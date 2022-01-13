import os
import numpy as np
import torch
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
from stsgcn.utils import get_model, read_config, get_optimizer, \
                         get_scheduler, get_data_loader, \
                         mpjpe_error, discriminator_loss, \
                         save_model, set_seeds, load_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# h36m constants
constant_joints = np.array([0, 1, 6, 11])
joints_to_be_imputed = np.array([16, 20, 23, 24, 28, 31])
joints_to_impute_with = np.array([13, 19, 22, 13, 27, 30])
constant_indices = np.concatenate([constant_joints * 3 + i for i in range(3)])
indices_to_be_imputed = np.concatenate([joints_to_be_imputed * 3 + i for i in range(3)])
indices_to_impute_with = np.concatenate([joints_to_impute_with * 3 + i for i in range(3)])
indices_to_predict = np.setdiff1d(np.arange(0, 96), np.concatenate([constant_indices, indices_to_be_imputed]))

# amass constants
joint_used = np.arange(4, 22)

def evaluation_epoch_amass(model, cfg, eval_data_loader, split):
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
            sequences_X = batch[:, 0:cfg["input_n"], joint_used, :]
            if split == 1:  # validation
                sequences_y = batch[:, cfg["input_n"]:cfg["input_n"]+cfg["output_n"], joint_used, :]
                sequences_yhat = model(sequences_X)  # (N, T, V, C)
                mpjpe_loss = mpjpe_error(sequences_yhat, sequences_y) * 1000
            elif split == 2:  # test
                sequences_y = batch[:, cfg["input_n"]:cfg["input_n"]+cfg["output_n"], full_joint_used, :]
                sequences_yhat_partial = model(sequences_X)  # (N, T, V, C)
                sequences_yhat_all = sequences_y.clone()
                sequences_yhat_all[:, :, joint_used, :] = sequences_yhat_partial
                mpjpe_loss = mpjpe_error(sequences_yhat_all, sequences_y) * 1000
            total_loss = mpjpe_loss
            eval_loss_dict.update({"mpjpe": mpjpe_loss.detach().cpu() * current_batch_size,
                                   "total": total_loss.detach().cpu() * current_batch_size})
    for loss_function, loss_value in eval_loss_dict.items():
        eval_loss_dict[loss_function] = loss_value / total_num_samples
    return eval_loss_dict

def evaluation_epoch_h36(model, cfg, eval_data_loader, split):
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
                sequences_X = batch[:, 0:cfg["input_n"], indices_to_predict].view(-1, cfg["input_n"], len(indices_to_predict) // 3, 3)  # (N, T, V, C)
                sequences_y = batch[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], indices_to_predict].view(-1, cfg["output_n"], len(indices_to_predict) // 3, 3)  # (N, T, V, C)
                sequences_yhat = model(sequences_X)  # (N, T, V, C)
                mpjpe_loss = mpjpe_error(sequences_yhat, sequences_y)
                total_loss = mpjpe_loss
            elif split == 2:  # test
                sequences_yhat_all = batch.clone()[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], :]
                sequences_X = batch[:, 0:cfg["input_n"], indices_to_predict].view(-1, cfg["input_n"], len(indices_to_predict) // 3, 3)
                sequences_y = batch[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], :].view(-1, cfg["output_n"], 32, 3)
                sequences_yhat_partial = model(sequences_X)  # (N, T, V, C)
                sequences_yhat_partial = sequences_yhat_partial.contiguous().view(-1, cfg["output_n"], len(indices_to_predict))
                sequences_yhat_all[:, :, indices_to_predict] = sequences_yhat_partial
                sequences_yhat_all[:, :, indices_to_be_imputed] = sequences_yhat_all[:, :, indices_to_impute_with]
                sequences_yhat_all = sequences_yhat_all.view(-1, cfg["output_n"], 32, 3)
                mpjpe_loss = mpjpe_error(sequences_yhat_all, sequences_y)
                total_loss = mpjpe_loss
            eval_loss_dict.update({"mpjpe": mpjpe_loss.detach().cpu() * current_batch_size,
                                   "total": total_loss.detach().cpu() * current_batch_size})
        return eval_loss_dict, total_num_samples_current_action

def evaluation_epoch(model, cfg, eval_data_loader, split):
    if cfg["dataset"] in ["amass_3d"]:
        eval_loss_dict = evaluation_epoch_amass(model, cfg, eval_data_loader, split)
    elif cfg["dataset"] in ["h36m_3d"]:
        if split == 1:  # validation
            eval_loss_dict, total_num_samples_current_action = evaluation_epoch_h36(model, cfg, eval_data_loader, split)
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
                eval_loss_dict_current_action, total_num_samples_current_action = evaluation_epoch_h36(model, cfg, current_eval_data_loader, split)
                total_num_samples += total_num_samples_current_action
                eval_loss_dict.update(eval_loss_dict_current_action)
                print(f"Evaluation loss for action '{action}': {eval_loss_dict_current_action['total'] / total_num_samples_current_action}")
            for loss_function, loss_value in eval_loss_dict.items():
                eval_loss_dict[loss_function] = loss_value / total_num_samples
    return eval_loss_dict

def test(config_path, args):
    cfg = read_config(config_path, args)
    set_seeds(cfg)
    test_data_loader = get_data_loader(cfg, split=2)
    gen_model = load_model(cfg, cfg["model_loc"]).to(device)
    
    test_loss_dict = evaluation_epoch(gen_model, cfg, test_data_loader, split=2)
    for loss_function, loss_value in test_loss_dict.items():
        logger.add_scalar(f"test/{loss_function}", loss_value, current_iteration)
    current_test_mpjpe_loss = test_loss_dict['mpjpe']
    print(f"Test mpjpe loss: {current_test_mpjpe_loss}")
