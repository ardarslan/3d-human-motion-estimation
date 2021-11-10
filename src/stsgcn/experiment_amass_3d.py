import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.autograd
from utils.experiment import Experiment
from stsgcn.model import Model
from utils.loss_funcs import mpjpe_error
from utils.amass_3d import Datasets

# from utils.dpw3d import * # choose amass or 3dpw by importing the right dataset class
from utils.amass_3d_viz import visualize as vis


class Amass3DExperiment(Experiment):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: %s" % self.device)
        self.model_name = "amass_3d_best_model"
        self.skip_rate = 5
        self.joints_to_consider = 18
        self.model = Model(
            self.cfg["input_dim"],
            self.cfg["input_n"],
            self.cfg["output_n"],
            self.cfg["st_gcnn_dropout"],
            self.joints_to_consider,
            self.cfg["n_tcnn_layers"],
            self.cfg["tcnn_kernel_size"],
            self.cfg["tcnn_dropout"],
        ).to(self.device)
        print(
            "total number of parameters of the network is: "
            + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        )

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg["lr"], weight_decay=1e-05)

        if self.cfg["use_scheduler"]:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.cfg["milestones"], gamma=self.cfg["gamma"]
            )

        train_loss = []
        val_loss = []
        self.model.train()
        Dataset = Datasets(
            data_dir=self.cfg["data_dir"],
            input_n=self.cfg["input_n"],
            output_n=self.cfg["output_n"],
            skip_rate=self.skip_rate,
            device=self.device,
            split=0,
        )

        loader_train = DataLoader(
            Dataset, batch_size=self.cfg["batch_size"], shuffle=True, num_workers=0
        )

        Dataset_val = Datasets(
            data_dir=self.cfg["data_dir"],
            input_n=self.cfg["input_n"],
            output_n=self.cfg["output_n"],
            skip_rate=self.skip_rate,
            device=self.device,
            split=1,
        )

        loader_val = DataLoader(
            Dataset_val, batch_size=self.cfg["batch_size"], shuffle=True, num_workers=0
        )
        joint_used = np.arange(4, 22)

        best_val_loss = np.inf
        for epoch in range(self.cfg["n_epochs"]):
            running_loss = 0
            n = 0
            self.model.train()
            for cnt, batch in enumerate(loader_train):
                batch = batch.float().to(self.device)[
                    :, :, joint_used
                ]  # multiply by 1000 for milimeters
                batch_dim = batch.shape[0]  # batch_size
                n += batch_dim

                sequences_train = batch[:, 0 : self.cfg["input_n"], :, :].permute(0, 3, 1, 2)
                sequences_predict_gt = batch[
                    :, self.cfg["input_n"] : self.cfg["input_n"] + self.cfg["output_n"], :, :
                ]
                optimizer.zero_grad()
                sequences_predict = self.model(sequences_train)
                loss = (
                    mpjpe_error(sequences_predict.permute(0, 1, 3, 2), sequences_predict_gt) * 1000
                )  # both must have format (batch, time, V, C)
                if cnt % 200 == 0:
                    print("[%d, %5d]  training loss: %.3f" % (epoch + 1, cnt + 1, loss.item()))
                loss.backward()
                if self.cfg["clip_grad"] is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["clip_grad"])
                optimizer.step()
                running_loss += loss * batch_dim
            train_loss.append(running_loss.detach().cpu() / n)
            self.model.eval()
            with torch.no_grad():
                running_loss = 0
                n = 0
                for cnt, batch in enumerate(loader_val):
                    batch = batch.float().to(self.device)[:, :, joint_used]
                    batch_dim = batch.shape[0]
                    n += batch_dim

                    sequences_train = batch[:, 0 : self.cfg["input_n"], :, :].permute(0, 3, 1, 2)
                    sequences_predict_gt = batch[
                        :, self.cfg["input_n"] : self.cfg["input_n"] + self.cfg["output_n"], :, :
                    ]
                    sequences_predict = self.model(sequences_train)
                    loss = (
                        mpjpe_error(sequences_predict.permute(0, 1, 3, 2), sequences_predict_gt)
                        * 1000
                    )  # the inputs to the loss function must have shape[N,T,V,C]
                    if cnt % 200 == 0:
                        print(
                            "[%d, %5d]  validation loss: %.3f" % (epoch + 1, cnt + 1, loss.item())
                        )
                    running_loss += loss * batch_dim
                current_val_loss = running_loss.detach().cpu() / n
                val_loss.append(current_val_loss)
            if self.cfg["use_scheduler"]:
                scheduler.step()

            if current_val_loss < best_val_loss:
                print("Saving the best model...")
                torch.save(
                    self.model.state_dict(), os.path.join(self.cfg["checkpoints_dir"], self.model_name)
                )

    def test(self):
        self.model.load_state_dict(
            torch.load(os.path.join(self.cfg['checkpoints_dir'], self.model_name))
        )
        self.model.eval()
        accum_loss = 0
        n = 0
        Dataset = Datasets(
            data_dir=self.cfg["data_dir"],
            input_n=self.cfg["input_n"],
            output_n=self.cfg["output_n"],
            skip_rate=self.skip_rate,
            device=self.device,
            split=2,
        )  # test
        loader_test = DataLoader(
            Dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            num_workers=0  # this means the thread executing the python code
            # will also prepare the data.
        )
        joint_used = np.arange(4, 22)  # we are using joints from 4 to 22.
        full_joint_used = np.arange(0, 22)  # needed for visualization
        with torch.no_grad():
            for cnt, batch in enumerate(loader_test):
                batch = batch.float().to(self.device)
                batch_dim = batch.shape[0]
                n += batch_dim

                sequences_train = batch[:, 0 : self.cfg["input_n"], joint_used, :].permute(
                    0, 3, 1, 2
                )

                sequences_predict_gt = batch[
                    :,
                    self.cfg["input_n"] : self.cfg["input_n"] + self.cfg["output_n"],
                    full_joint_used,
                    :,
                ]

                sequences_predict = self.model(sequences_train).permute(0, 1, 3, 2)

                all_joints_seq = sequences_predict_gt.clone()

                all_joints_seq[:, :, joint_used, :] = sequences_predict

                loss = (
                    mpjpe_error(all_joints_seq, sequences_predict_gt) * 1000
                )  # loss in milimeters
                accum_loss += loss * batch_dim
        print("overall average loss in mm is: " + str(accum_loss / n))

    def visualize(self):
        self.model.load_state_dict(
            torch.load(os.path.join(self.cfg['checkpoints_dir'], self.model_name))
        )
        self.model.eval()
        vis(
            self.cfg["input_n"],
            self.cfg["output_n"],
            self.cfg["visualize_from"],
            self.cfg["data_dir"],
            self.model,
            self.device,
            self.cfg["n_viz"],
            self.skip_rate,
        )
