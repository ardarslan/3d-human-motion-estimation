import os
import random
import torch
import numpy as np
import time
import yaml
import shutil
from stsgcn.models import ZeroVelocity, STSGCN
from gnngru.model import GNNGRU
from stsgcn.datasets import H36M_3D_Dataset, H36M_Ang_Dataset, Amass_3D_Dataset, DPW_3D_Dataset
from torch.utils.data import DataLoader

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_model(cfg):#done
    if cfg["model"] == "zero_velocity":
        model = ZeroVelocity(cfg)
    elif cfg["model"] == "stsgcn":
        model = STSGCN(cfg)
    elif cfg["model"] == "gnngru":
        model = GNNGRU(cfg)
        model.apply(weights_init)
    else:
        raise Exception("Not implemented yet.")
    
      
    print(
        "Total number of parameters: "
        + str(sum(p.numel() for p in model.parameters() if p.requires_grad))
    )
    return model

    
def get_optimizer(cfg, model):#done
    print(model.parameters())
    if cfg["optimizer"] == "adam":
        optimizer = optim.Adam(params = model.parameters(), 
                            lr =cfg["lr"], 
                            weight_decay=cfg["weight_decay"])
    else:
        raise Exception("Not implemented yet.")


def get_scheduler(cfg, optimizer):#done
    if cfg["scheduler"] == "multi_step_lr":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg["milestones"], gamma=cfg["gamma"]
        )
    else:
        raise Exception("Not implemented yet.")


def get_data_loader(cfg, split, actions=None):#done
    if cfg["dataset"] == "amass_3d":
        Dataset = Amass_3D_Dataset
    elif cfg["dataset"] == "h36m_3d":
        Dataset = H36M_3D_Dataset

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
        pin_memory=True
    )

    return data_loader

def mpjpe_error(batch_pred, batch_gt): #done from loss_l2
    batch_pred = batch_pred.contiguous().view(-1, 3)
    batch_gt = batch_gt.contiguous().view(-1, 3)
    return torch.mean(torch.norm(batch_gt - batch_pred, 2, 1))
    
    
    
def read_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["experiment_time"] = str(int(time.time()))
    os.makedirs(os.path.join(cfg["log_dir"], cfg["experiment_time"]), exist_ok=True)
    config_file_name = config_path.split("/")[-1]
    shutil.copyfile(config_path, os.path.join(cfg["log_dir"], cfg["experiment_time"], config_file_name))

    if cfg["dataset"] == "amass_3d":
        cfg["joints_to_consider"] = 18
        cfg["skip_rate"] = 1
        cfg["loss_function"] = "mpjpe"
    elif cfg["dataset"] == "h36m_3d":
        cfg["joints_to_consider"] = 22 #look what to do!!!!!
        cfg["skip_rate"] = 5
        cfg["loss_function"] = "mpjpe"
    else:
        raise Exception("Not a valid dataset.")
    return cfg

    

def save_model(model, cfg):
    print("Saving the best model...")
    checkpoints_dir = os.path.join(cfg["log_dir"], cfg["experiment_time"])
    torch.save(
        model.state_dict(), os.path.join(checkpoints_dir, "best_model")
    )


def load_model(cfg):
    checkpoints_dir = os.path.join(cfg["log_dir"], cfg["experiment_time"])
    model = get_model(cfg)
    model.load_state_dict(torch.load(os.path.join(checkpoints_dir, "best_model")))
    return model


def set_seeds(cfg):
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
