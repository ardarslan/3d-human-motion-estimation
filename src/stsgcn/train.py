import numpy as np
import torch
from stsgcn.utils import get_model, read_config, get_optimizer, \
                         get_scheduler, get_data_loader, \
                         mpjpe_error, save_model, set_seeds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_step_amass(model, optimizer, cfg, train_data_loader):
    joint_used = np.arange(4, 22)
    total_train_loss = 0
    total_num_samples = 0
    model.train()
    for batch in train_data_loader:
        batch = batch.float().to(device)[:, :, joint_used, :]  # (N, T, V, C)
        current_batch_size = batch.shape[0]
        total_num_samples += current_batch_size
        sequences_X = batch[:, 0:cfg["input_n"], :, :]  # (N, T, V, C)
        sequences_y = batch[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], :, :]  # (N, T, V, C)
        optimizer.zero_grad()
        sequences_yhat = model(sequences_X)  # (N, T, V, C)
        train_loss = mpjpe_error(sequences_yhat, sequences_y) * 1000
        train_loss.backward()
        if cfg["clip_grad"] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
        optimizer.step()
        total_train_loss += train_loss * current_batch_size
    average_train_loss = total_train_loss.detach().cpu() / total_num_samples
    return average_train_loss


def evaluation_step_amass(model, cfg, eval_data_loader, split):
    joint_used = np.arange(4, 22)
    full_joint_used = np.arange(0, 22)
    model.eval()
    with torch.no_grad():
        total_eval_loss = 0
        total_num_samples = 0
        for batch in eval_data_loader:
            batch = batch.float().to(device)  # (N, T, V, C)
            current_batch_size = batch.shape[0]
            total_num_samples += current_batch_size
            sequences_X = batch[:, 0:cfg["input_n"], joint_used, :]
            if split == 1:  # validation
                sequences_y = batch[:, cfg["input_n"]:cfg["input_n"]+cfg["output_n"], joint_used, :]
                sequences_yhat = model(sequences_X)
                eval_loss = mpjpe_error(sequences_yhat, sequences_y) * 1000
            elif split == 2:  # test
                sequences_y = batch[:, cfg["input_n"]:cfg["input_n"]+cfg["output_n"], full_joint_used, :]
                sequences_yhat_partial = model(sequences_X)
                sequences_yhat_all = sequences_y.clone()
                sequences_yhat_all[:, :, joint_used, :] = sequences_yhat_partial
                eval_loss = mpjpe_error(sequences_yhat_all, sequences_y) * 1000
            total_eval_loss += eval_loss * current_batch_size
        average_eval_loss = total_eval_loss.detach().cpu() / total_num_samples
    return average_eval_loss


def train_step_h36(model, optimizer, cfg, train_data_loader):
    constant_joints = np.array([0, 1, 6, 11])
    joints_to_be_imputed = np.array([16, 20, 23, 24, 28, 31])
    joints_to_impute_with = np.array([13, 19, 22, 13, 27, 30])

    constant_indices = np.concatenate([constant_joints * 3 + i for i in range(3)])
    indices_to_be_imputed = np.concatenate([joints_to_be_imputed * 3 + i for i in range(3)])
    indices_to_impute_with = np.concatenate([joints_to_impute_with * 3 + i for i in range(3)])

    indices_to_predict = np.setdiff1d(np.arange(0, 96), np.concatenate([constant_indices, indices_to_be_imputed]))

    total_train_loss = 0
    total_num_samples = 0
    model.train()
    for batch in train_data_loader:
        batch = batch.float().to(device)  # (N, T, V, C)
        current_batch_size = batch.shape[0]
        total_num_samples += current_batch_size
        sequences_X = batch[:, 0:cfg["input_n"], indices_to_predict].view(-1, cfg["input_n"], len(indices_to_predict) // 3, 3)  # (N, T, V, C)
        sequences_y = batch[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], indices_to_predict].view(-1, cfg["output_n"], len(indices_to_predict) // 3, 3)  # (N, T, V, C)
        optimizer.zero_grad()
        sequences_yhat = model(sequences_X)  # (N, T, V, C)
        train_loss = mpjpe_error(sequences_yhat, sequences_y)
        train_loss.backward()
        if cfg["clip_grad"] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
        optimizer.step()
        total_train_loss += train_loss * current_batch_size
    average_train_loss = total_train_loss.detach().cpu() / total_num_samples
    return average_train_loss


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
        total_evaluation_loss_current_action = 0
        total_num_samples_current_action = 0

        for batch in eval_data_loader:
            batch = batch.float().to(device)
            current_batch_size = batch.shape[0]
            total_num_samples_current_action += current_batch_size
            if split == 1:  # validation
                sequences_X = batch[:, 0:cfg["input_n"], indices_to_predict].view(-1, cfg["input_n"], len(indices_to_predict) // 3, 3)  # (N, T, V, C)
                sequences_y = batch[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], indices_to_predict].view(-1, cfg["output_n"], len(indices_to_predict) // 3, 3)  # (N, T, V, C)
                sequences_yhat = model(sequences_X)  # (N, T, V, C)
                evaluation_loss = mpjpe_error(sequences_yhat, sequences_y)
            elif split == 2:  # test
                sequences_yhat_all = batch.clone()[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], :]
                sequences_X = batch[:, 0:cfg["input_n"], indices_to_predict].view(-1, cfg["input_n"], len(indices_to_predict) // 3, 3)
                sequences_y = batch[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], :].view(-1, cfg["output_n"], 32, 3)
                sequences_yhat_partial = model(sequences_X).contiguous().view(-1, cfg["output_n"], len(indices_to_predict))
                sequences_yhat_all[:, :, indices_to_predict] = sequences_yhat_partial
                sequences_yhat_all[:, :, indices_to_be_imputed] = sequences_yhat_all[:, :, indices_to_impute_with]
                sequences_yhat_all = sequences_yhat_all.view(-1, cfg["output_n"], 32, 3)
                evaluation_loss = mpjpe_error(sequences_yhat_all, sequences_y)
            total_evaluation_loss_current_action += evaluation_loss * current_batch_size

        return total_num_samples_current_action, total_evaluation_loss_current_action


def train_step(model, optimizer, cfg, train_data_loader):
    if cfg["dataset"] in ["amass_3d"]:
        average_train_loss = train_step_amass(model, optimizer, cfg, train_data_loader)
    elif cfg["dataset"] in ["h36m_3d"]:
        average_train_loss = train_step_h36(model, optimizer, cfg, train_data_loader)
    return average_train_loss


def evaluation_step(model, cfg, eval_data_loader, split):
    if cfg["dataset"] in ["amass_3d"]:
        average_eval_loss = evaluation_step_amass(model, cfg, eval_data_loader, split)
    elif cfg["dataset"] in ["h36m_3d"]:
        if split == 1:  # validation
            total_num_samples, total_evaluation_loss = evaluation_step_h36(model, cfg, eval_data_loader, split)
            average_eval_loss = total_evaluation_loss / total_num_samples
        elif split == 2:  # test
            actions = ["walking", "eating", "smoking", "discussion", "directions",
                       "greeting", "phoning", "posing", "purchases", "sitting",
                       "sittingdown", "takingphoto", "waiting", "walkingdog",
                       "walkingtogether"]
            total_num_samples = 0
            total_evaluation_loss = 0
            for action in actions:
                current_eval_data_loader = get_data_loader(cfg, split=2, actions=[action])
                total_num_samples_current_action, total_evaluation_loss_current_action = evaluation_step_h36(model, cfg, current_eval_data_loader, split)
                total_num_samples += total_num_samples_current_action
                total_evaluation_loss += total_evaluation_loss_current_action
                print(f"Evaluation loss for action '{action}': {total_evaluation_loss_current_action / total_num_samples_current_action}")
            average_eval_loss = total_evaluation_loss / total_num_samples
    return average_eval_loss


def train(config_path):
    cfg = read_config(config_path)
    set_seeds(cfg)

    model = get_model(cfg).to(device)
    optimizer = get_optimizer(cfg, model)

    if cfg["use_scheduler"]:
        scheduler = get_scheduler(cfg, optimizer)

    train_data_loader = get_data_loader(cfg, split=0)
    validation_data_loader = get_data_loader(cfg, split=1)

    train_losses = []
    validation_losses = []

    best_validation_loss = np.inf
    for epoch in range(cfg["n_epochs"]):
        # train
        current_train_loss = train_step(model, optimizer, cfg, train_data_loader)
        train_losses.append(train_losses)

        # validate
        current_validation_loss = evaluation_step(model, cfg, validation_data_loader, split=1)
        validation_losses.append(current_validation_loss)

        print(f"Epoch: {epoch}. Train loss: {current_train_loss}. Validation loss: {current_validation_loss}")

        if cfg["use_scheduler"]:
            scheduler.step()

        if current_validation_loss < best_validation_loss:
            save_model(model, cfg)

    # test
    test_data_loader = get_data_loader(cfg, split=2)
    test_loss = evaluation_step(model, cfg, test_data_loader, split=2)
    print(f"Test loss: {test_loss}")


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
