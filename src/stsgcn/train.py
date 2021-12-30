import os
import numpy as np
import torch
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
from stsgcn.utils import get_model, read_config, get_optimizer, \
                         get_scheduler, get_data_loader, \
                         mpjpe_error, save_model, set_seeds, \
                         load_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_step_amass(model, optimizer, cfg, train_data_loader):
    joint_used = np.arange(4, 22)
    total_num_samples = 0
    model.train()
    train_loss_dict = Counter()
    for batch in train_data_loader:
        batch = batch.float().to(device)[:, :, joint_used, :]  # (N, T, V, C)
        current_batch_size = batch.shape[0]
        total_num_samples += current_batch_size
        sequences_X = batch[:, 0:cfg["input_n"], :, :]  # (N, T, V, C)
        sequences_y = batch[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], :, :]  # (N, T, V, C)
        optimizer.zero_grad()
        sequences_yhat = model(sequences_X)  # (N, T, V, C)
        mpjpe_loss = mpjpe_error(sequences_yhat, sequences_y) * 1000
        total_loss = mpjpe_loss
        total_loss.backward()
        train_loss_dict.update({"mpjpe": mpjpe_loss.detach().cpu() * current_batch_size,
                                "total": total_loss.detach().cpu() * current_batch_size})
        if cfg["clip_grad"] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
        optimizer.step()
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
            sequences_X = batch[:, 0:cfg["input_n"], joint_used, :]
            if split == 1:  # validation
                sequences_y = batch[:, cfg["input_n"]:cfg["input_n"]+cfg["output_n"], joint_used, :]
                sequences_yhat = model(sequences_X)
                mpjpe_loss = mpjpe_error(sequences_yhat, sequences_y) * 1000
            elif split == 2:  # test
                sequences_y = batch[:, cfg["input_n"]:cfg["input_n"]+cfg["output_n"], full_joint_used, :]
                sequences_yhat_partial = model(sequences_X)
                sequences_yhat_all = sequences_y.clone()
                sequences_yhat_all[:, :, joint_used, :] = sequences_yhat_partial
                mpjpe_loss = mpjpe_error(sequences_yhat_all, sequences_y) * 1000
            total_loss = mpjpe_loss
            eval_loss_dict.update({"mpjpe": mpjpe_loss.detach().cpu() * current_batch_size,
                                   "total": total_loss.detach().cpu() * current_batch_size})
    for loss_function, loss_value in eval_loss_dict.items():
        eval_loss_dict[loss_function] = loss_value / total_num_samples
    return eval_loss_dict


def train_step_h36(model, optimizer, cfg, train_data_loader):
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
    for batch in train_data_loader:
        batch = batch.float().to(device)  # (N, T, V, C)
        current_batch_size = batch.shape[0]
        total_num_samples += current_batch_size
        sequences_X = batch[:, 0:cfg["input_n"], indices_to_predict].view(-1, cfg["input_n"], len(indices_to_predict) // 3, 3)  # (N, T, V, C)
        sequences_y = batch[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], indices_to_predict].view(-1, cfg["output_n"], len(indices_to_predict) // 3, 3)  # (N, T, V, C)
        optimizer.zero_grad()

        # specific to model type stsgcn_transformer
        if cfg['model'] == 'stsgcn_transformer':
            tgt = sequences_y.transpose(2,3).reshape(sequences_X.shape[0],cfg['output_n'],-1).to(device)
            tgt_mask = torch.triu(torch.ones(cfg['output_n'],cfg['output_n']) == 1, diagonal = 1).to(device)
            sequences_yhat = model(sequences_X,tgt,tgt_mask)  # (N, T, V, C)
        else: # for any other model
            sequences_yhat = model(sequences_X)  # (N, T, V, C)
            
        mpjpe_loss = mpjpe_error(sequences_yhat, sequences_y)
        total_loss = mpjpe_loss
        total_loss.backward()
        train_loss_dict.update({"mpjpe": mpjpe_loss.detach().cpu() * current_batch_size,
                                "total": total_loss.detach().cpu() * current_batch_size})
        if cfg["clip_grad"] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
        optimizer.step()
    for loss_function, loss_value in train_loss_dict.items():
        train_loss_dict[loss_function] = loss_value / total_num_samples
    return train_loss_dict


def evaluation_step_h36(model, cfg, eval_data_loader, split):
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
        eval_loss_dict = {}
        eval_loss_dict_valid = Counter()
        for batch in eval_data_loader:
            batch = batch.float().to(device)
            current_batch_size = batch.shape[0]
            total_num_samples_current_action += current_batch_size
            if split == 1:  # validation
                sequences_X = batch[:, 0:cfg["input_n"], indices_to_predict].view(-1, cfg["input_n"], len(indices_to_predict) // 3, 3)  # (N, T, V, C)
                sequences_y = batch[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], indices_to_predict].view(-1, cfg["output_n"], len(indices_to_predict) // 3, 3)  # (N, T, V, C)
                
                # specific to model type stsgcn_transformer, do it autoregressive manner
                if cfg['model'] == 'stsgcn_transformer':
                    #sos token
                    src = sequences_X.transpose(2,3).reshape(sequences_X.shape[0],cfg['input_n'],-1).to(device)
                    tgt = src[:,-1:,:]
                    tgt_mask = torch.triu(torch.ones(1,1) == 1, diagonal = 1).to(device)

                    for _ in range(cfg["output_n"]):

                        temp_pred = model(sequences_X,tgt,tgt_mask)  # (N, T, V, C)
                        temp_pred = temp_pred.transpose(2,3).reshape(sequences_X.shape[0],tgt.shape[1],-1).to(device)
                        # take the last element of prediction and append to the already predicted ones.
                        tgt = torch.cat((tgt,temp_pred[:,-1:,:]),dim=1).to(device)
                        tgt_mask = torch.triu(torch.ones(tgt.shape[1],tgt.shape[1]) == 1, diagonal = 1).to(device)

                    pred = tgt[:,1:,:]
                    pred = pred.view(tgt.shape[0],cfg["output_n"],cfg['input_dim'],-1)
                    sequences_yhat = pred.permute(0, 1, 3, 2)
                else:
                    sequences_yhat = model(sequences_X)  # (N, T, V, C)
                    
                mpjpe_loss = mpjpe_error(sequences_yhat, sequences_y)
                total_loss = mpjpe_loss
                eval_loss_dict_valid.update({"mpjpe": mpjpe_loss.detach().cpu() * current_batch_size,
                                   "total": total_loss.detach().cpu() * current_batch_size})
            elif split == 2:  # test
                sequences_yhat_all = batch.clone()[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], :]
                sequences_X = batch[:, 0:cfg["input_n"], indices_to_predict].view(-1, cfg["input_n"], len(indices_to_predict) // 3, 3)
                sequences_y = batch[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], :].view(-1, cfg["output_n"], 32, 3)
                
                # specific to model type stsgcn_transformer
                if cfg['model'] == 'stsgcn_transformer':
                    #sos token
                    src = sequences_X.transpose(2,3).reshape(sequences_X.shape[0],cfg['input_n'],-1).to(device)
                    tgt = src[:,-1:,:]
                    tgt_mask = torch.triu(torch.ones(1,1) == 1, diagonal = 1).to(device)

                    for _ in range(cfg["output_n"]):

                        temp_pred = model(sequences_X,tgt,tgt_mask)  # (N, T, V, C)
                        temp_pred = temp_pred.transpose(2,3).reshape(sequences_X.shape[0],tgt.shape[1],-1).to(device)
                        tgt = torch.cat((tgt,temp_pred[:,-1:,:]),dim=1).to(device)
                        tgt_mask = torch.triu(torch.ones(tgt.shape[1],tgt.shape[1]) == 1, diagonal = 1).to(device)

                    pred = tgt[:,1:,:]
                    pred = pred.view(tgt.shape[0],cfg["output_n"],cfg['input_dim'],-1)
                    sequences_yhat = pred.permute(0, 1, 3, 2)
                else:
                    sequences_yhat = model(sequences_X)  # (N, T, V, C)
                
                
                sequences_yhat_partial = sequences_yhat.contiguous().view(-1, cfg["output_n"], len(indices_to_predict))
                sequences_yhat_all[:, :, indices_to_predict] = sequences_yhat_partial
                sequences_yhat_all[:, :, indices_to_be_imputed] = sequences_yhat_all[:, :, indices_to_impute_with]
                sequences_yhat_all = sequences_yhat_all.view(-1, cfg["output_n"], 32, 3)
                for output_n in [2,10,25]:#[2,4,8,10,14,18,22,25]: # 80,160,320,400,560,720,880,1000 msec
                    mpjpe_loss = mpjpe_error(sequences_yhat_all[:,:output_n,:,:], sequences_y[:,:output_n,:,:])
                    total_loss = mpjpe_loss
                    eval_loss_dict[output_n] = total_loss.detach().cpu() * current_batch_size

        if split == 1:
            return eval_loss_dict_valid,total_num_samples_current_action
        else:
            return eval_loss_dict, total_num_samples_current_action



def train_step(model, optimizer, cfg, train_data_loader):
    if cfg["dataset"] in ["amass_3d"]:
        train_loss_dict = train_step_amass(model, optimizer, cfg, train_data_loader)
    elif cfg["dataset"] in ["h36m_3d"]:
        train_loss_dict = train_step_h36(model, optimizer, cfg, train_data_loader)
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
                print(f"{action :<15}-> " , end='')
                for output_n in [2,10,25]:#[2,4,8,10,14,18,22,25]: # 80,160,320,400,560,720,880,1000 msec
                    pr_loss= "{:.1f}".format(eval_loss_dict_current_action[output_n] / total_num_samples_current_action)
                    print(f"{output_n * 40}ms: {pr_loss} | ", end='')
                print('')
            for msec, loss_value in eval_loss_dict.items():
                eval_loss_dict[msec] = loss_value / total_num_samples
    return eval_loss_dict


def train(config_path):
    cfg = read_config(config_path)
    set_seeds(cfg)

    model = get_model(cfg).to(device)
    print(cfg['model'], cfg['input_n'], cfg['output_n'])
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
        train_loss_dict = train_step(model, optimizer, cfg, train_data_loader)
        for loss_function, loss_value in train_loss_dict.items():
            logger.add_scalar(f"train/{loss_function}", loss_value, epoch)
        current_total_train_loss = train_loss_dict['total']

        # validate
        if cfg['model'] != 'stsgcn_transformer' or epoch % 3 == 0:
            validation_loss_dict = evaluation_step(model, cfg, validation_data_loader, split=1)
            for loss_function, loss_value in validation_loss_dict.items():
                logger.add_scalar(f"validation/{loss_function}", loss_value, epoch)
            current_total_validation_loss = validation_loss_dict['total']

            print(f"Epoch: {epoch} Train loss: {current_total_train_loss} Validation loss: {current_total_validation_loss}")

            if current_total_validation_loss < best_validation_loss:
                save_model(model, cfg)
                best_validation_loss = current_total_validation_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter == cfg["early_stop_patience"]:
                break
        else:
            print(f"Epoch: {epoch} Train loss: {current_total_train_loss}")
            
        if cfg["use_scheduler"]:
            scheduler.step()
        
    # test
    test_data_loader = get_data_loader(cfg, split=2)
    model = load_model(cfg).to(device)
    test_loss_dict = evaluation_step(model, cfg, test_data_loader, split=2)
    #for loss_function, loss_value in test_loss_dict.items():
    #    logger.add_scalar(f"test/{loss_function}", loss_value, 1)

    print(f"-------\n{'Average' :<15}->", end='')
    for output_n in [2,10,25]:#[2,4,8,10,14,18,22,25]: # 80,160,320,400,560,720,880,1000 msec
        pr_loss= "{:.1f}".format(test_loss_dict[output_n])
        print(f"{output_n * 40}ms: {pr_loss} | ", end='')
    print('\n-------')


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
