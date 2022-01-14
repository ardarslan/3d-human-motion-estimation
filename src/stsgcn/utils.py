import os
import random
import torch
import numpy as np
import time
import yaml
import shutil
import torch.nn as nn
from torch.nn import functional as F
from stsgcn.models import ZeroVelocity, STSGCN, MotionDiscriminator, RNN_STSEncoder, STSGCN_Transformer, SimpleRNN
from stsgcn.datasets import H36M_3D_Dataset, H36M_Ang_Dataset, Amass_3D_Dataset, DPW_3D_Dataset
from torch.utils.data import DataLoader
from pprint import pprint


def get_model(cfg, model_type):
    model_name_model_mapping = {
        "zero_velocity": ZeroVelocity,
        "stsgcn": STSGCN,
        "motion_disc": MotionDiscriminator,
        "rnn_stsE": RNN_STSEncoder,
        "stsgcn_transformer": STSGCN_Transformer,
        "simple_rnn": SimpleRNN
    }

    if model_type == "gen":
        model = model_name_model_mapping[cfg["gen_model"]](cfg)
    elif model_type == "disc":
        model = model_name_model_mapping[cfg["disc_model"]](cfg)
    else:
        print("Valid Models = ", model_name_model_mapping.keys())
        raise Exception("Not a valid model type.")

    print(
        f"Total number of parameters in {model_type} model: "
        + str(sum(p.numel() for p in model.parameters() if p.requires_grad))
    )
    return model


def get_optimizer(cfg, model, model_type):
    if model_type == "gen":
        return torch.optim.Adam(model.parameters(), lr=cfg["gen_lr"], weight_decay=cfg["gen_weight_decay"])
    elif model_type == "disc":
        return torch.optim.Adam(model.parameters(), lr=cfg["disc_lr"], weight_decay=cfg["disc_weight_decay"])
    else:
        raise Exception("Not a valid model type.")


def get_scheduler(cfg, optimizer, model_type):
    scheduler_type = cfg[f"{model_type}_scheduler"]
    if scheduler_type == "multi_step_lr":
        # n_epochs = cfg["n_epochs"]
        # milestone_interval = int(n_epochs / 5)
        if model_type == "gen":
            gamma = cfg["gen_gamma"]
            milestones = cfg["gen_milestones"]
        elif model_type == "disc":
            gamma = cfg["disc_gamma"]
            milestones = cfg["disc_milestones"]
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
    elif scheduler_type == "step_lr":
        if model_type == "gen":
            gamma = cfg["gen_gamma"]
        elif model_type == "disc":
            gamma = cfg["disc_gamma"]
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=gamma
        )
    # elif cfg["scheduler"] == "lambda":
    #     def lambda_rule(epoch):
    #         lr_l = 1.0 - max(0, epoch - cfg["n_epoch_fix"]) / float(cfg["n_epochs"] - cfg["n_epoch_fix"] + 1)
    #         return lr_l
    #     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        raise Exception("Not implemented yet.")


def get_data_loader(cfg, split, actions=None):
    if cfg["dataset"] == "amass_3d":
        Dataset = Amass_3D_Dataset
    elif cfg["dataset"] == "h36m_3d":
        Dataset = H36M_3D_Dataset
    # elif cfg["dataset"] == "h36m_ang":
    #     Dataset = H36M_Ang_Dataset
    # elif cfg["dataset"] == "dpw_3d":
    #     Dataset = DPW_3D_Dataset
    else:
        raise Exception("Not a valid dataset.")

    dataset = Dataset(data_dir=cfg["data_dir"],
                      input_n=cfg["input_n"],
                      output_n=cfg["output_n"],
                      skip_rate=cfg["skip_rate"],
                      body_model_dir=cfg["body_model_dir"],
                      actions=actions,
                      split=split)

    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=(split != 2),
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True
    )
    return data_loader


def mpjpe_error(batch_pred, batch_gt):
    batch_pred = batch_pred.contiguous().view(-1, 3)
    batch_gt = batch_gt.contiguous().view(-1, 3)
    return torch.mean(torch.norm(batch_gt - batch_pred, 2, 1))


def mojo_loss(X, Y_r, Y, mu, logvar, cfg):
    MSE = F.l1_loss(Y_r, Y) + cfg["lambda_tf"] * F.l1_loss(Y_r[1:]-Y_r[:-1], Y[1:]-Y[:-1])
    MSE_v = F.l1_loss(X[-1], Y_r[0])
    KLD = 0.5 * torch.mean(-1 - logvar + mu.pow(2) + logvar.exp())
    if cfg["robustkl"]:
        KLD = torch.sqrt(1 + KLD**2)-1
    loss_r = MSE + cfg["lambda_v"] * MSE_v + cfg["beta"] * KLD
    return loss_r  # , np.array([loss_r.item(), MSE.item(), MSE_v.item(), KLD.item()])


def discriminator_loss(y, yhat):
    criterion = nn.BCELoss()
    return criterion(yhat, y)


def read_config(config_path, args):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["experiment_time"] = str(int(time.time()))
    os.makedirs(os.path.join(cfg["log_dir"], cfg["experiment_time"]), exist_ok=True)
    config_file_name = config_path.split("/")[-1]
    shutil.copyfile(config_path, os.path.join(cfg["log_dir"], cfg["experiment_time"], config_file_name))

    cfg["data_dir"] = args.data_dir
    cfg["gen_model"] = args.gen_model
    cfg["dataset"] = args.dataset
    cfg["output_n"] = args.output_n
    cfg["gen_clip_grad"] = args.gen_clip_grad
    cfg["recurrent_cell"] = args.recurrent_cell
    cfg["batch_size"] = args.batch_size
    cfg["use_disc"] = args.use_disc
    cfg["gen_lr"] = args.gen_lr
    cfg["early_stop_patience"] = args.early_stop_patience
    cfg["gen_gamma"] = args.gen_gamma
    cfg["gen_milestones"] = args.gen_milestones

    if cfg["dataset"] == "amass_3d":
        cfg["joints_to_consider"] = 18
        cfg["skip_rate"] = 1
        cfg["nx"] = 54
        cfg["ny"] = 54
    elif cfg["dataset"] == "h36m_3d":
        cfg["joints_to_consider"] = 22
        cfg["skip_rate"] = 5
        cfg["nx"] = 66
        cfg["ny"] = 66
    else:
        raise Exception("Not a valid dataset.")

    pprint(cfg)
    return cfg


def save_model(model, cfg):
    print("Saving the best model...")
    checkpoints_dir = os.path.join(cfg["log_dir"], cfg["experiment_time"])
    torch.save(model.state_dict(), os.path.join(checkpoints_dir, "best_model"))


def load_model(cfg, loc=None):
    if loc is not None:
        checkpoints_dir = cfg["model_loc"]
    else:    
        checkpoints_dir = os.path.join(cfg["log_dir"], cfg["experiment_time"])
    model = get_model(cfg, "gen")
    model.load_state_dict(torch.load(os.path.join(checkpoints_dir, "best_model")))
    return model


def set_seeds(cfg):
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
