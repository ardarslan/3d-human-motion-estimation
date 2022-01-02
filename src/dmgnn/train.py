import os
import numpy as np
import torch
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
from gnngru.utils import get_model, read_config, get_optimizer, \
                         get_scheduler, get_data_loader, \
                         mpjpe_error, save_model, set_seeds, \
                         load_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#Giving encoder, decoder and target params
def train_sample(data_set, batch_size, source_seq_len, target_seq_len, input_size, cfg):
    #all_keys = list(data_set.keys())
    #chosen_keys_idx = np.random.choice(len(all_keys), batch_size)
    total_seq_len = source_seq_len + target_seq_len

    encoder_inputs  = np.zeros((batch_size, source_seq_len-1, input_size), dtype=np.float32)
    decoder_inputs  = np.zeros((batch_size, 1, input_size), dtype=np.float32)
    decoder_outputs = np.zeros((batch_size, target_seq_len, input_size), dtype=np.float32)

    for i in range(batch_size):
        the_key = all_keys[chosen_keys_idx[i]]
        t, d = data_set[the_key].shape
        idx = np.random.randint(16, t-total_seq_len)
        data_sel = data_set[the_key][idx:idx+total_seq_len,:]

        encoder_inputs[i,:,:]  = data_sel[0:source_seq_len-1,:]
        decoder_inputs[i,:,:]  = data_sel[source_seq_len-1:source_seq_len,:]
        decoder_outputs[i,:,:] = data_sel[source_seq_len:,:]

    #rs = int(np.random.uniform(low=0, high=4))
    #downsample_idx = np.array([int(i)+rs for i in [np.floor(j*4) for j in range(12)]])

    return encoder_inputs, decoder_inputs, decoder_outputs, downsample_idx
    
  
    

#train by step, #cfg contains yaml file
def train_step_h36(model, optimizer, cfg, train_data_loader):
    constant_joints = np.array([0, 1, 6, 11])
    joints_to_be_imputed = np.array([16, 21, 22, 23, 24, 29, 30, 31])
    #joints_to_be_imputed = np.array([16, 20, 23, 24, 28, 31])
    joints_to_impute_with = np.array([13, 19, 25, 13, 27, 32])
    #joints_to_impute_with = np.array([13, 19, 22, 13, 27, 30])

    constant_indices = np.concatenate([constant_joints * 3 + i for i in range(3)])
    indices_to_be_imputed = np.concatenate([joints_to_be_imputed * 3 + i for i in range(3)])
    indices_to_impute_with = np.concatenate([joints_to_impute_with * 3 + i for i in range(3)])

    indices_to_predict = np.setdiff1d(np.arange(0, 96), np.concatenate([constant_indices, indices_to_be_imputed]))

    #check where to get data_set_dict from
    #see where to get batch_size, source_seq_len, input_size from
    
    total_num_samples = 0
    model.train()
    train_loss_dict = Counter()
    for batch in train_data_loader:
        batch = batch.float().to(device)  # (N, T, V, C) N=Batchsize, V=joints, T = time, C = 3
        current_batch_size = batch.shape[0]
        total_num_samples += current_batch_size
        optimizer.zero_grad()
        """
        sequences_X = batch[:, 0:cfg["input_n"], indices_to_predict].view(-1, cfg["input_n"], len(indices_to_predict) // 3, 3)  # (N, T, V, C)
        sequences_y = batch[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], indices_to_predict].view(-1, cfg["output_n"], len(indices_to_predict) // 3, 3)  # (N, T, V, C)
        sequences_X = sequences_X.permute(0, 2, 1, 3) # (N, T, V, C) => (N, V, T, C)
        sequences_y = sequences_y.permute(0, 2, 1, 3) # (N, T, V, C) => (N, V, T, C)
        """
        #check dim_to_use = indices_to_predict, train_dict = train_data_loader(try) !!!  
        """
        encoder_inputs, decoder_inputs, targets, downsample_idx = train_sample(batch, 
                                                               current_batch_size, 
                                                               cfg["input_n"], 
                                                               cfg["output_n"], 
                                                               len(indices_to_predict), cfg)
        """
        encoder_inputs = batch[:, 0:cfg["input_n"]-1, indices_to_predict]  # (N, T, V*C)
        decoder_inputs = batch[:, cfg["input_n"]-1, indices_to_predict]  # (N, T, V*C)
        targets = batch[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], indices_to_predict]  # (N, T, V*C)
        
        #do encoder
        encoder_inputs_v = np.zeros_like(encoder_inputs)
        encoder_inputs_v[:, 1:, :] = encoder_inputs[:, 1:, :]-encoder_inputs[:, :-1, :]
        encoder_inputs_a = np.zeros_like(encoder_inputs)
        encoder_inputs_a[:, :-1, :] = encoder_inputs_v[:, 1:, :]-encoder_inputs_v[:, :-1, :]

        encoder_inputs_p = torch.Tensor(encoder_inputs).float()
        encoder_inputs_v = torch.Tensor(encoder_inputs_v).float()
        encoder_inputs_a = torch.Tensor(encoder_inputs_a).float()
        
        #do decoder 
        decoder_inputs = torch.Tensor(decoder_inputs).float()
        decoder_inputs_previous = torch.Tensor(encoder_inputs[:, -1, :]).unsqueeze(1)
        decoder_inputs_previous2 = torch.Tensor(encoder_inputs[:, -2, :]).unsqueeze(1)
        targets = torch.Tensor(targets).float()  # [N,T,D] = [64, 10, 63]
        #downsample_idx = torch.Tensor(downsample_idx).long()

        N, T, D = targets.size()  # N = 64(batchsize), T=10, D=63
        targets = targets.contiguous().view(N, T, -1, 3).permute(0, 2, 1, 3)  # [64, 21, 10, 3]
        
        sequences_yhat = model(encoder_inputs_p,
                             encoder_inputs_v,
                             encoder_inputs_a,
                             decoder_inputs,
                             decoder_inputs_previous,
                             decoder_inputs_previous2,
                             cfg["output_n"])#, layout = "h36m")  # (N, V, T, C)
        mpjpe_loss = mpjpe_error(sequences_yhat, targets)
        total_loss = mpjpe_loss
        total_loss.backward()
        train_loss_dict.update({"mpjpe": mpjpe_loss.detach().cpu() * current_batch_size,
                                "total": total_loss.detach().cpu() * current_batch_size})
        if cfg["clip_grad"] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
        optimizer.step()
    for loss_function, loss_value in train_loss_dict.items():
        train_loss_dict[loss_function] = loss_value / total_num_samples
    print("T_N_Samples = ", total_num_samples)
    return train_loss_dict


def evaluation_step_h36(model, cfg, eval_data_loader, split):

    model.eval()

    constant_joints = np.array([0, 1, 6, 11])
    joints_to_be_imputed = np.array([16, 21, 22, 23, 24, 29, 30, 31])
    #joints_to_be_imputed = np.array([16, 20, 23, 24, 28, 31])
    joints_to_impute_with = np.array([13, 19, 25, 13, 27, 32])
    #joints_to_impute_with = np.array([13, 19, 22, 13, 27, 30])

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
                """
                sequences_X = batch[:, 0:cfg["input_n"], indices_to_predict].view(-1, cfg["input_n"], len(indices_to_predict) // 3, 3)  # (N, T, V, C)
                sequences_y = batch[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], indices_to_predict].view(-1, cfg["output_n"], len(indices_to_predict) // 3, 3)  # (N, T, V, C)
                sequences_X = sequences_X.permute(0, 2, 1, 3) # (N, T, V, C) => (N, V, T, C)
                sequences_y = sequences_y.permute(0, 2, 1, 3) # (N, T, V, C) => (N, V, T, C)
                sequences_yhat = model(sequences_X)  # (N, V, T, C)
                """
                encoder_inputs = batch[:, 0:cfg["input_n"]-1, indices_to_predict]  # (N, T, V*C)
                decoder_inputs = batch[:, cfg["input_n"]-1, indices_to_predict]  # (N, T, V*C)
                targets = batch[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], indices_to_predict]  # (N, T, V*C)
                
                #do encoder
                encoder_inputs_v = np.zeros_like(encoder_inputs)
                encoder_inputs_v[:, 1:, :] = encoder_inputs[:, 1:, :]-encoder_inputs[:, :-1, :]
                encoder_inputs_a = np.zeros_like(encoder_inputs)
                encoder_inputs_a[:, :-1, :] = encoder_inputs_v[:, 1:, :]-encoder_inputs_v[:, :-1, :]

                encoder_inputs_p = torch.Tensor(encoder_inputs).float()
                encoder_inputs_v = torch.Tensor(encoder_inputs_v).float()
                encoder_inputs_a = torch.Tensor(encoder_inputs_a).float()
                
                #do decoder 
                decoder_inputs = torch.Tensor(decoder_inputs).float()
                decoder_inputs_previous = torch.Tensor(encoder_inputs[:, -1, :]).unsqueeze(1)
                decoder_inputs_previous2 = torch.Tensor(encoder_inputs[:, -2, :]).unsqueeze(1)
                targets = torch.Tensor(targets).float()  # [N,T,D] = [64, 10, 63]
                #downsample_idx = torch.Tensor(downsample_idx).long()

                N, T, D = targets.size()  # N = 64(batchsize), T=10, D=63
                targets = targets.contiguous().view(N, T, -1, 3).permute(0, 2, 1, 3)  # [64, 21, 10, 3]
                
                sequences_yhat = model(encoder_inputs_p,
                                     encoder_inputs_v,
                                     encoder_inputs_a,
                                     decoder_inputs,
                                     decoder_inputs_previous,
                                     decoder_inputs_previous2,
                                     cfg["output_n"])#, layout = "h36m")  # (N, V, T, C)
                
                mpjpe_loss = mpjpe_error(sequences_yhat, targets)
                total_loss = mpjpe_loss
            elif split == 2:  # test #!!!!!!!!!!!!
                sequences_yhat_all = batch.clone()[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], :]
                """
                sequences_X = batch[:, 0:cfg["input_n"], indices_to_predict].view(-1, cfg["input_n"], len(indices_to_predict) // 3, 3)
                sequences_y = batch[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], :].view(-1, cfg["output_n"], 32, 3)
                sequences_X = sequences_X.permute(0, 2, 1, 3) # (N, T, V, C) => (N, V, T, C)
                sequences_y = sequences_y.permute(0, 2, 1, 3) # (N, T, V, C) => (N, V, T, C)
                """
                encoder_inputs = batch[:, 0:cfg["input_n"]-1, indices_to_predict]  # (N, T, V*C)
                decoder_inputs = batch[:, cfg["input_n"]-1, indices_to_predict]  # (N, T, V*C)
                targets = batch[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], :]  # (N, T, V*C)
                
                #do encoder
                encoder_inputs_v = np.zeros_like(encoder_inputs)
                encoder_inputs_v[:, 1:, :] = encoder_inputs[:, 1:, :]-encoder_inputs[:, :-1, :]
                encoder_inputs_a = np.zeros_like(encoder_inputs)
                encoder_inputs_a[:, :-1, :] = encoder_inputs_v[:, 1:, :]-encoder_inputs_v[:, :-1, :]

                encoder_inputs_p = torch.Tensor(encoder_inputs).float()
                encoder_inputs_v = torch.Tensor(encoder_inputs_v).float()
                encoder_inputs_a = torch.Tensor(encoder_inputs_a).float()
                
                #do decoder 
                decoder_inputs = torch.Tensor(decoder_inputs).float()
                decoder_inputs_previous = torch.Tensor(encoder_inputs[:, -1, :]).unsqueeze(1)
                decoder_inputs_previous2 = torch.Tensor(encoder_inputs[:, -2, :]).unsqueeze(1)
                targets = torch.Tensor(targets).float()  # [N,T,D] = [64, 10, 63]
                #downsample_idx = torch.Tensor(downsample_idx).long()

                N, T, D = targets.size()  # N = 64(batchsize), T=10, D=63
                targets = targets.contiguous().view(N, T, -1, 3).permute(0, 2, 1, 3)  # [64, 21, 10, 3]
                
                output = model(encoder_inputs_p,
                                     encoder_inputs_v,
                                     encoder_inputs_a,
                                     decoder_inputs,
                                     decoder_inputs_previous,
                                     decoder_inputs_previous2,
                                     cfg["output_n"])#, layout = "h36m")  # (N, V, T, C)
                
                sequences_yhat_partial = output.contiguous().view(-1, cfg["output_n"], len(indices_to_predict))
                sequences_yhat_all[:, :, indices_to_predict] = sequences_yhat_partial
                sequences_yhat_all[:, :, indices_to_be_imputed] = sequences_yhat_all[:, :, indices_to_impute_with]
                sequences_yhat_all = sequences_yhat_all.view(-1, cfg["output_n"], 32, 3)
                mpjpe_loss = mpjpe_error(sequences_yhat_all, targets)
                total_loss = mpjpe_loss
            eval_loss_dict.update({"mpjpe": mpjpe_loss.detach().cpu() * current_batch_size,
                                   "total": total_loss.detach().cpu() * current_batch_size})
        # for loss_function, loss_value in eval_loss_dict.items():
        #     eval_loss_dict[loss_function] = loss_value / total_num_samples_current_action
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
        train_loss_dict = train_step(model, optimizer, cfg, train_data_loader)
        for loss_function, loss_value in train_loss_dict.items():
            logger.add_scalar(f"train/{loss_function}", loss_value, epoch)
        current_total_train_loss = train_loss_dict['total']

        # validate
        validation_loss_dict = evaluation_step(model, cfg, validation_data_loader, split=1)
        for loss_function, loss_value in validation_loss_dict.items():
            logger.add_scalar(f"validation/{loss_function}", loss_value, epoch)
        current_total_validation_loss = validation_loss_dict['total']

        print(f"Epoch: {epoch}. Train loss: {current_total_train_loss}. Validation loss: {current_total_validation_loss}")

        if cfg["use_scheduler"]:
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