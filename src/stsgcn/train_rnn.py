import os
import numpy as np
import torch
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
from stsgcn.utils import get_model, read_config, get_optimizer, \
                         get_scheduler, get_data_loader, \
                         mpjpe_error, pose_joint_sum_error, \
                         save_model, set_seeds, load_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_step_amass(model, optimizer, scheduler, cfg, train_data_loader):
    joint_used = np.arange(4, 22)
    total_num_samples = 0
    model.train()
    train_loss_dict = Counter()
    for idx, batch in enumerate(train_data_loader):
        batch = batch.float().to(device)[:, :, joint_used, :]  # (N, T, V, C)
        current_batch_size = batch.shape[0]
        total_num_samples += current_batch_size
        
        sequences_X = batch[:, :cfg["input_n"] + cfg["output_n"], :, :]  # (N, T, V, C)
        sequences_y = batch[:, 1:cfg["input_n"] + cfg["output_n"]+1, :, :]  # (N, T, V, C)
        optimizer.zero_grad()
        sequences_yhat = model(sequences_X)  # (N, T, V, C)
        
        if cfg["loss_type"] is None or cfg["loss_type"] == "mpjpe":
            mpjpe_loss = mpjpe_error(sequences_yhat, sequences_y) * 1000
            total_loss = mpjpe_loss
            train_loss_dict.update({"mpjpe": mpjpe_loss.detach().cpu() * current_batch_size,
                                    "total": total_loss.detach().cpu() * current_batch_size})
            
        elif cfg["loss_type"] == "pose_joint_sum":
            per_joint_loss = pose_joint_sum_error(sequences_yhat, sequences_y, seq_len=cfg["input_n"]+cfg["output_n"], num_joints=joint_used.shape, dim_per_joint=3)
            total_loss = per_joint_loss
            train_loss_dict.update({"per_joint_loss": per_joint_loss.detach().cpu() * current_batch_size,
                                    "total": total_loss.detach().cpu() * current_batch_size})
        else:
            raise Exception("Unknown Loss Type")
        total_loss.backward()
        if cfg["clip_grad"] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
        optimizer.step()
        if cfg["use_scheduler"] and cfg["decay_every_steps"] is not None and cfg["decay_every_steps"] and idx % 1000 == 0:
            scheduler.step()
    for loss_function, loss_value in train_loss_dict.items():
        train_loss_dict[loss_function] = loss_value / total_num_samples
    return train_loss_dict


def evaluation_step_amass(model, cfg, eval_data_loader, split):
    joint_used = np.arange(4, 22)
    full_joint_used = np.arange(0, 22)
    model.eval()
    with torch.no_grad():
        total_num_samples = 0
        eval_loss_dict = Counter()
        for batch in eval_data_loader:
            batch = batch.float().to(device)  # (N, T, V, C)
            current_batch_size = batch.shape[0]
            total_num_samples += current_batch_size
            
            sequences_X = batch[:, :cfg["input_n"], joint_used, :]
            if split == 1:  # validation
                sequences_y = batch[:, cfg["input_n"]:cfg["input_n"]+cfg["output_n"], joint_used, :]
                sequences_yhat = model(sequences_X)
                if cfg["loss_type"] is None or cfg["loss_type"] == "mpjpe":
                    mpjpe_loss = mpjpe_error(sequences_yhat, sequences_y) * 1000
                    total_loss = mpjpe_loss
                elif cfg["loss_type"] == "pose_joint_sum":
                    per_joint_loss = pose_joint_sum_error(sequences_yhat, sequences_y, seq_len=cfg["output_n"], num_joints=joint_used.shape, dim_per_joint=3)
                    total_loss = per_joint_loss
                else:
                    raise Exception("Unknown Loss Type")
            elif split == 2:  # test
                sequences_y = batch[:, cfg["input_n"]:cfg["input_n"]+cfg["output_n"], full_joint_used, :]
                sequences_yhat_partial = model(sequences_X)
                sequences_yhat_all = sequences_y.clone()
                sequences_yhat_all[:, :, joint_used, :] = sequences_yhat_partial
                if cfg["loss_type"] is None or cfg["loss_type"] == "mpjpe":
                    mpjpe_loss = mpjpe_error(sequences_yhat_all, sequences_y) * 1000
                    total_loss = mpjpe_loss
                elif cfg["loss_type"] == "pose_joint_sum":
                    per_joint_loss = pose_joint_sum_error(sequences_yhat_all, sequences_y, seq_len=cfg["output_n"], num_joints=full_joint_used.shape, dim_per_joint=3)
                    total_loss = per_joint_loss
                else:
                    raise Exception("Unknown Loss Type")
            if cfg["loss_type"] is None or cfg["loss_type"] == "mpjpe":
                eval_loss_dict.update({"mpjpe": mpjpe_loss.detach().cpu() * current_batch_size,
                                       "total": total_loss.detach().cpu() * current_batch_size})
            elif cfg["loss_type"] == "pose_joint_sum":
                eval_loss_dict.update({"per_joint_loss": per_joint_loss.detach().cpu() * current_batch_size,
                                       "total": total_loss.detach().cpu() * current_batch_size})
            else:
                raise Exception("Unknown Loss Type")
    for loss_function, loss_value in eval_loss_dict.items():
        eval_loss_dict[loss_function] = loss_value / total_num_samples
    return eval_loss_dict


def train_step_h36(model, optimizer, scheduler, cfg, train_data_loader):
    constant_joints = np.array([0, 1, 6, 11])
    joints_to_be_imputed = np.array([16, 20, 23, 24, 28, 31])
    joints_to_impute_with = np.array([13, 19, 22, 13, 27, 30])

    constant_indices = np.concatenate([constant_joints * 3 + i for i in range(3)])
    indices_to_be_imputed = np.concatenate([joints_to_be_imputed * 3 + i for i in range(3)])
    indices_to_impute_with = np.concatenate([joints_to_impute_with * 3 + i for i in range(3)])

    indices_to_predict = np.setdiff1d(np.arange(0, 96), np.concatenate([constant_indices, indices_to_be_imputed]))

    total_num_samples = 0
    model.train()
    train_loss_dict = Counter()
    for idx, batch in enumerate(train_data_loader):
        batch = batch.float().to(device)  # (N, T, V, C)
        current_batch_size = batch.shape[0]
        total_num_samples += current_batch_size
        
        sequences_X = batch[:, :cfg["input_n"] + cfg["output_n"], indices_to_predict].view(
                            -1, cfg["input_n"] + cfg["output_n"], len(indices_to_predict) // 3, 3)  # (N, T, V, C)
        sequences_y = batch[:, 1:cfg["input_n"] + cfg["output_n"]+1, indices_to_predict].view(
                            -1, cfg["input_n"] + cfg["output_n"], len(indices_to_predict) // 3, 3)  # (N, T, V, C)
        
        optimizer.zero_grad()
        sequences_yhat = model(sequences_X)  # (N, T, V, C)
        if cfg["loss_type"] is None or cfg["loss_type"] == "mpjpe":
            mpjpe_loss = mpjpe_error(sequences_yhat, sequences_y) * 1000
            total_loss = mpjpe_loss
            train_loss_dict.update({"mpjpe": mpjpe_loss.detach().cpu() * current_batch_size,
                                    "total": total_loss.detach().cpu() * current_batch_size})
            
        elif cfg["loss_type"] == "pose_joint_sum":
            per_joint_loss = pose_joint_sum_error(sequences_yhat, sequences_y, seq_len=cfg["output_n"], num_joints=len(indices_to_predict)//3, dim_per_joint=3)
            total_loss = per_joint_loss
            train_loss_dict.update({"per_joint_loss": per_joint_loss.detach().cpu() * current_batch_size,
                                    "total": total_loss.detach().cpu() * current_batch_size})
            
        else:
            raise Exception("Unknown Loss Type")
        total_loss.backward()
        if cfg["clip_grad"] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
        optimizer.step()
        if cfg["use_scheduler"] and cfg["decay_every_steps"] is not None and cfg["decay_every_steps"] and idx % 1000 == 0:
            scheduler.step()
    for loss_function, loss_value in train_loss_dict.items():
        train_loss_dict[loss_function] = loss_value / total_num_samples
    return train_loss_dict


def evaluation_step_h36(model, cfg, eval_data_loader, split):
    """
    Unlike other step methods, this one returns sum of losses
    in loss_dict. Not the average.
    """

    model.eval()

    constant_joints = np.array([0, 1, 6, 11])
    joints_to_be_imputed = np.array([16, 20, 23, 24, 28, 31])
    joints_to_impute_with = np.array([13, 19, 22, 13, 27, 30])

    constant_indices = np.concatenate([constant_joints * 3 + i for i in range(3)])
    indices_to_be_imputed = np.concatenate([joints_to_be_imputed * 3 + i for i in range(3)])
    indices_to_impute_with = np.concatenate([joints_to_impute_with * 3 + i for i in range(3)])

    indices_to_predict = np.setdiff1d(np.arange(0, 96), np.concatenate([constant_indices, indices_to_be_imputed]))

    with torch.no_grad():
        total_num_samples_current_action = 0
        eval_loss_dict = Counter()
        for batch in eval_data_loader:
            batch = batch.float().to(device)
            current_batch_size = batch.shape[0]
            total_num_samples_current_action += current_batch_size
            if split == 1:  # validation
                sequences_X = batch[:, :cfg["input_n"], indices_to_predict].view(-1, cfg["input_n"], len(indices_to_predict) // 3, 3)  # (N, T, V, C)
                sequences_y = batch[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], indices_to_predict].view(-1, cfg["output_n"], len(indices_to_predict) // 3, 3)  # (N, T, V, C)
                sequences_yhat = model(sequences_X)  # (N, T, V, C)
                
                if cfg["loss_type"] is None or cfg["loss_type"] == "mpjpe":
                    mpjpe_loss = mpjpe_error(sequences_yhat, sequences_y) * 1000
                    total_loss = mpjpe_loss
                elif cfg["loss_type"] == "pose_joint_sum":
                    per_joint_loss = pose_joint_sum_error(sequences_yhat, sequences_y, seq_len=cfg["output_n"], num_joints=len(indices_to_predict)//3, dim_per_joint=3)
                    total_loss = per_joint_loss
                else:
                    raise Exception("Unknown Loss Type")
            elif split == 2:  # test
                sequences_yhat_all = batch.clone()[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], :]
                sequences_X = batch[:, :cfg["input_n"], indices_to_predict].view(-1, cfg["input_n"], len(indices_to_predict) // 3, 3)
                sequences_y = batch[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], :].view(-1, cfg["output_n"], 32, 3)
                
                sequences_yhat_partial = model(sequences_X).contiguous().view(-1, cfg["output_n"], len(indices_to_predict))
                sequences_yhat_all[:, :, indices_to_predict] = sequences_yhat_partial
                sequences_yhat_all[:, :, indices_to_be_imputed] = sequences_yhat_all[:, :, indices_to_impute_with]
                sequences_yhat_all = sequences_yhat_all.view(-1, cfg["output_n"], 32, 3)
                
                if cfg["loss_type"] is None or cfg["loss_type"] == "mpjpe":
                    mpjpe_loss = mpjpe_error(sequences_yhat_all, sequences_y) * 1000
                    total_loss = mpjpe_loss
                elif cfg["loss_type"] == "pose_joint_sum":
                    per_joint_loss = pose_joint_sum_error(sequences_yhat_all, sequences_y, seq_len=cfg["output_n"], num_joints=32, dim_per_joint=3)
                    total_loss = per_joint_loss
                else:
                    raise Exception("Unknown Loss Type")
            if cfg["loss_type"] is None or cfg["loss_type"] == "mpjpe":
                eval_loss_dict.update({"mpjpe": mpjpe_loss.detach().cpu() * current_batch_size,
                                       "total": total_loss.detach().cpu() * current_batch_size})
            elif cfg["loss_type"] == "pose_joint_sum":
                eval_loss_dict.update({"per_joint_loss": per_joint_loss.detach().cpu() * current_batch_size,
                                       "total": total_loss.detach().cpu() * current_batch_size})
            else:
                raise Exception("Unknown Loss Type")
        # for loss_function, loss_value in eval_loss_dict.items():
        #     eval_loss_dict[loss_function] = loss_value / total_num_samples_current_action
        return eval_loss_dict, total_num_samples_current_action


def train_step(model, optimizer, scheduler, cfg, train_data_loader):
    if cfg["dataset"] in ["amass_3d"]:
        train_loss_dict = train_step_amass(model, optimizer, scheduler, cfg, train_data_loader)
    elif cfg["dataset"] in ["h36m_3d"]:
        train_loss_dict = train_step_h36(model, optimizer, scheduler, cfg, train_data_loader)
    return train_loss_dict


def evaluation_step(model, cfg, eval_data_loader, split):
    if cfg["dataset"] in ["amass_3d"]:
        eval_loss_dict = evaluation_step_amass(model, cfg, eval_data_loader, split)
    elif cfg["dataset"] in ["h36m_3d"]:
        if split == 1:  # validation
            eval_loss_dict, total_num_samples_current_action = evaluation_step_h36(model, cfg, eval_data_loader, split)
            for loss_function, loss_value in eval_loss_dict.items():
                eval_loss_dict[loss_function] = loss_value / total_num_samples_current_action
        elif split == 2:  # test
            actions = ["walking", "eating", "smoking", "discussion", "directions",
                       "greeting", "phoning", "posing", "purchases", "sitting",
                       "sittingdown", "takingphoto", "waiting", "walkingdog",
                       "walkingtogether"]
            total_num_samples = 0
            eval_loss_dict = Counter()
            for action in actions:
                current_eval_data_loader = get_data_loader(cfg, split=2, actions=[action])
                eval_loss_dict_current_action, total_num_samples_current_action = evaluation_step_h36(model, cfg, current_eval_data_loader, split)
                total_num_samples += total_num_samples_current_action
                eval_loss_dict.update(eval_loss_dict_current_action)
                print(f"Evaluation loss for action '{action}': {eval_loss_dict_current_action['total'] / total_num_samples_current_action}")
            for loss_function, loss_value in eval_loss_dict.items():
                eval_loss_dict[loss_function] = loss_value / total_num_samples
    return eval_loss_dict


def train(config_path):
    cfg = read_config(config_path)
    set_seeds(cfg)

    model = get_model(cfg).to(device)
    optimizer = get_optimizer(cfg, model)

    if cfg["use_scheduler"]:
        scheduler = get_scheduler(cfg, optimizer)

    train_data_loader = get_data_loader(cfg, split=0)
    validation_data_loader = get_data_loader(cfg, split=1)

    logger = SummaryWriter(os.path.join(cfg["log_dir"], cfg["experiment_time"]))

    best_validation_loss = np.inf
    early_stop_counter = 0
    for epoch in range(cfg["n_epochs"]):
        # train
        train_loss_dict = train_step(model, optimizer, scheduler, cfg, train_data_loader)
        for loss_function, loss_value in train_loss_dict.items():
            logger.add_scalar(f"train/{loss_function}", loss_value, epoch)
        current_total_train_loss = train_loss_dict['total']

        # validate
        validation_loss_dict = evaluation_step(model, cfg, validation_data_loader, split=1)
        for loss_function, loss_value in validation_loss_dict.items():
            logger.add_scalar(f"validation/{loss_function}", loss_value, epoch)
        current_total_validation_loss = validation_loss_dict['total']

        print(f"Epoch: {epoch}. Train loss: {current_total_train_loss}. Validation loss: {current_total_validation_loss}")

        if cfg["use_scheduler"] and (cfg["decay_every_steps"] is None or not cfg["decay_every_steps"]):
            scheduler.step()

        if current_total_validation_loss < best_validation_loss:
            save_model(model, cfg)
            best_validation_loss = current_total_validation_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter == cfg["early_stop_patience"]:
            break

    # test
    test_data_loader = get_data_loader(cfg, split=2)
    model = load_model(cfg).to(device)
    test_loss_dict = evaluation_step(model, cfg, test_data_loader, split=2)
    for loss_function, loss_value in test_loss_dict.items():
        logger.add_scalar(f"test/{loss_function}", loss_value, 1)
    current_total_test_loss = test_loss_dict['total']
    print(f"Test loss: {current_total_test_loss}")


# def visualize(self):
#     self.model.load_state_dict(
#         torch.load(os.path.join(self.cfg['checkpoints_dir'], self.model_name))
#     )
#     self.model.eval()
#     vis(
#         self.cfg["input_n"],
#         self.cfg["output_n"],
#         self.cfg["visualize_from"],
#         self.cfg["data_dir"],
#         self.model,
#         self.device,
#         self.cfg["n_viz"],
#         self.skip_rate,
#         self.cfg["body_model_dir"]
#     )
