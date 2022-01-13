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


def train_step(gen_model, disc_model, gen_optimizer, disc_optimizer, gen_batch, disc_batch, epoch, cfg):
    if cfg["dataset"] == "amass_3d":
        disc_sequences_X = disc_batch[:, 0:cfg["input_n"], joint_used, :]  # (N, T, V, C)
        disc_sequences_real_y = disc_batch[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], joint_used, :]  # (N, T, V, C)
        gen_sequences_X = gen_batch[:, 0:cfg["input_n"], joint_used, :]  # (N, T, V, C)
        if cfg["gen_model"] == "simple_rnn":
            gen_sequences_real_y = gen_batch[:, 1:cfg["input_n"] + cfg["output_n"], joint_used, :]  # (N, T, V, C)
        else:
            gen_sequences_real_y = gen_batch[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], joint_used, :]  # (N, T, V, C)
    elif cfg["dataset"] == "h36m_3d":
        disc_sequences_X = disc_batch[:, 0:cfg["input_n"], indices_to_predict].view(-1, cfg["input_n"], len(indices_to_predict) // 3, 3)  # (N, T, V, C)
        disc_sequences_real_y = disc_batch[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], indices_to_predict].view(-1, cfg["output_n"], len(indices_to_predict) // 3, 3)  # (N, T, V, C)
        gen_sequences_X = gen_batch[:, 0:cfg["input_n"], indices_to_predict].view(-1, cfg["input_n"], len(indices_to_predict) // 3, 3)  # (N, T, V, C)
        if cfg["gen_model"] == "simple_rnn":
            gen_sequences_real_y = gen_batch[:, 1:cfg["input_n"] + cfg["output_n"], indices_to_predict].view(-1, cfg["input_n"]+cfg["output_n"]-1, len(indices_to_predict) // 3, 3)  # (N, T, V, C)
        else:
            gen_sequences_real_y = gen_batch[:, cfg["input_n"]:cfg["input_n"] + cfg["output_n"], indices_to_predict].view(-1, cfg["output_n"], len(indices_to_predict) // 3, 3)  # (N, T, V, C)
    else:
        raise Exception("Not a valid dataset.")

    # train discriminator
    if cfg["use_disc"] and epoch >= cfg["start_training_discriminator_epoch"]:
        disc_model.train()
        gen_model.eval()
        disc_model.zero_grad()
        gen_model.zero_grad()
        disc_optimizer.zero_grad()
        gen_optimizer.zero_grad()

        with torch.no_grad():
            disc_sequences_generated_y = gen_model(disc_sequences_X).detach().contiguous()  # (N, T, V, C)

        disc_prediction_on_real_y = disc_model(disc_sequences_real_y)
        disc_prediction_on_generated_y = disc_model(disc_sequences_generated_y)

        disc_preds_on_gen_samples = disc_prediction_on_generated_y.mean()
        disc_preds_on_real_samples = disc_prediction_on_real_y.mean()

        # coeff = cfg["batch_size"] * cfg["disc_gaussian_std_step_size"]
        # disc_gaussian_std_adjust = coeff * np.sign(disc_preds_on_real_samples.detach().cpu().numpy() - cfg["disc_real_prediction_target"])
        # disc_model.gaussian_noise_std = np.maximum(disc_model.gaussian_noise_std + disc_gaussian_std_adjust, 0)

        disc_loss_for_real_y = discriminator_loss(0.8 * torch.ones_like(disc_prediction_on_real_y, dtype=torch.float, device=device), disc_prediction_on_real_y)
        disc_loss_for_generated_y = discriminator_loss(0.2 * torch.ones_like(disc_prediction_on_generated_y, dtype=torch.float, device=device), disc_prediction_on_generated_y)
        disc_total_loss = disc_loss_for_real_y + disc_loss_for_generated_y
        disc_total_loss.backward()
        if cfg["disc_clip_grad"] is not None:
            torch.nn.utils.clip_grad_norm_(disc_model.parameters(), cfg["disc_clip_grad"])
        disc_optimizer.step()

    # train generator
    gen_model.train()
    gen_model.zero_grad()
    gen_optimizer.zero_grad()
    gen_sequences_yhat = gen_model(gen_sequences_X)  # (N, T, V, C)
    gen_mpjpe_loss = mpjpe_error(gen_sequences_yhat, gen_sequences_real_y) * 1000

    if cfg["use_disc"] and epoch >= cfg["start_training_discriminator_epoch"]:
        disc_model.train()
        disc_model.zero_grad()
        disc_optimizer.zero_grad()

        disc_sequences_generated_y = gen_model(disc_sequences_X).contiguous()  # (N, T, V, C)

        disc_prediction_on_generated_y = disc_model(disc_sequences_generated_y)
        gen_disc_loss = cfg["gen_disc_loss_weight"] * discriminator_loss(0.8 * torch.ones_like(disc_prediction_on_generated_y, dtype=torch.float, device=device), disc_prediction_on_generated_y)
        if epoch >= cfg["start_feeding_discriminator_loss_epoch"]:
            gen_total_loss = gen_mpjpe_loss + gen_disc_loss
        else:
            gen_total_loss = gen_mpjpe_loss
    else:
        gen_total_loss = gen_mpjpe_loss

    gen_total_loss.backward()
    if cfg["gen_clip_grad"] is not None:
        torch.nn.utils.clip_grad_norm_(gen_model.parameters(), cfg["gen_clip_grad"])
    gen_optimizer.step()

    train_loss_dict = {"gen_mpjpe": gen_mpjpe_loss.detach().cpu(),
                       "gen_total": gen_total_loss.detach().cpu()}

    if cfg["use_disc"] and epoch >= cfg["start_training_discriminator_epoch"]:
        train_loss_dict.update({
            "gen_disc": gen_disc_loss.detach().cpu(),
            "disc_real": disc_loss_for_real_y.detach().cpu(),
            "disc_gen": disc_loss_for_generated_y.detach().cpu(),
            "disc_total": disc_total_loss.detach().cpu(),
            "disc_preds_on_gen_samples": disc_preds_on_gen_samples.detach().cpu(),
            "disc_preds_on_real_samples": disc_preds_on_real_samples.detach().cpu(),
           #  "disc_gaussian_noise_std": disc_model.gaussian_noise_std
        })
    return train_loss_dict


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


def train(config_path, args):
    cfg = read_config(config_path, args)
    set_seeds(cfg)

    gen_model = get_model(cfg, model_type="gen").to(device)
    gen_optimizer = get_optimizer(cfg, gen_model, "gen")

    disc_model = get_model(cfg, model_type="disc").to(device)
    disc_optimizer = get_optimizer(cfg, disc_model, "disc")

    if cfg["use_scheduler"]:
        gen_scheduler = get_scheduler(cfg, gen_optimizer, "gen")
        disc_scheduler = get_scheduler(cfg, disc_optimizer, "disc")

    disc_train_data_loader = get_data_loader(cfg, split=0)
    gen_train_data_loader = get_data_loader(cfg, split=0)
    validation_data_loader = get_data_loader(cfg, split=1)

    logger = SummaryWriter(os.path.join(cfg["log_dir"], cfg["experiment_time"]))

    best_validation_loss = np.inf
    early_stop_counter = 0
    current_iteration = 0
    for epoch in range(cfg["n_epochs"]):
        gen_train_data_loader_iter = iter(gen_train_data_loader)
        # train
        for disc_batch in disc_train_data_loader:
            disc_batch = disc_batch.float().to(device)
            gen_batch = next(gen_train_data_loader_iter).float().to(device)
            current_iteration += 1
            train_loss_dict = train_step(gen_model, disc_model, gen_optimizer, disc_optimizer, gen_batch, disc_batch, epoch, cfg)
            for loss_function, loss_value in train_loss_dict.items():
                logger.add_scalar(f"train/{loss_function}", loss_value, current_iteration)
            if current_iteration % cfg["print_train_loss_every_iter"] == 0:
                print(f"Epoch: {epoch}, Iter: {current_iteration}, " + ", ".join([f"{loss_function}: {loss_value}" for loss_function, loss_value in train_loss_dict.items()]))

        # validate
        validation_loss_dict = evaluation_epoch(gen_model, cfg, validation_data_loader, split=1)
        for loss_function, loss_value in validation_loss_dict.items():
            logger.add_scalar(f"validation/{loss_function}", loss_value, current_iteration)
        current_validation_mpjpe_loss = validation_loss_dict['mpjpe']

        print(f"Epoch: {epoch+1}, Iter: {current_iteration}, Validation mpjpe loss: {current_validation_mpjpe_loss}")

        if cfg["use_scheduler"]:
            gen_scheduler.step()
            if cfg["use_disc"] and epoch >= cfg["start_training_discriminator_epoch"]:
                disc_scheduler.step()

        if current_validation_mpjpe_loss < best_validation_loss:
            save_model(gen_model, cfg)
            best_validation_loss = current_validation_mpjpe_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter == cfg["early_stop_patience"]:
            break

    # test
    test_data_loader = get_data_loader(cfg, split=2)
    gen_model = load_model(cfg).to(device)
    test_loss_dict = evaluation_epoch(gen_model, cfg, test_data_loader, split=2)
    for loss_function, loss_value in test_loss_dict.items():
        logger.add_scalar(f"test/{loss_function}", loss_value, current_iteration)
    current_test_mpjpe_loss = test_loss_dict['mpjpe']
    print(f"Test mpjpe loss: {current_test_mpjpe_loss}")


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


# def train(config_path):
#     cfg = read_config(config_path)
#     set_seeds(cfg)
#     train_data_loader = get_data_loader(cfg, 0)

#     device = "cuda"

#     model = get_model(cfg).to(device)
#     optimizer = get_optimizer(cfg, model)
#     scheduler = get_scheduler(cfg, optimizer)

#     model.train()
#     for epoch in range(0, cfg["n_epochs"]):
#         train_losses = 0
#         total_num_sample = 0
#         loss_names = ['TOTAL', 'MSE', 'MSE_v', 'KLD']
#         constant_joints = np.array([0, 1, 6, 11])
#         joints_to_be_imputed = np.array([16, 20, 23, 24, 28, 31])
#         joints_to_impute_with = np.array([13, 19, 22, 13, 27, 30])

#         constant_indices = np.concatenate([constant_joints * 3 + i for i in range(3)])
#         indices_to_be_imputed = np.concatenate([joints_to_be_imputed * 3 + i for i in range(3)])
#         indices_to_impute_with = np.concatenate([joints_to_impute_with * 3 + i for i in range(3)])

#         indices_to_predict = np.setdiff1d(np.arange(0, 96), np.concatenate([constant_indices, indices_to_be_imputed]))

#         input_n = 15
#         output_n = 45

#         for batch in train_data_loader:
#             batch = batch.float().to(device)  # (N, T, V, C)
#             current_batch_size = batch.shape[0]
#             total_num_sample += current_batch_size
#             X = batch[:, 0:input_n, indices_to_predict].view(-1, input_n, len(indices_to_predict) // 3, 3)  # (N, T, V, C)
#             Y = batch[:, input_n:input_n + output_n, indices_to_predict].view(-1, output_n, len(indices_to_predict) // 3, 3)  # (N, T, V, C)
#             Y_r, mu, logvar = model(X, Y)
#             X = X.permute(1, 0, 2, 3)  # (T, N, V, C)
#             X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3])  # (T, N, V*C)
#             Y = Y.permute(1, 0, 2, 3)  # (T, N, V, C)
#             Y = Y.reshape(Y.shape[0], Y.shape[1], Y.shape[2]*Y.shape[3])  # (T, N, V*C)
#             loss = mojo_loss(X, Y_r, Y, mu, logvar, cfg)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             train_losses += loss * current_batch_size

#         scheduler.step()
#         train_losses /= total_num_sample
#         print('====> Epoch: {} Loss: {}'.format(epoch, train_losses))
