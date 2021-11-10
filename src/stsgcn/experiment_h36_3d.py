import os
from utils import h36motion3d as datasets
from torch.utils.data import DataLoader
from stsgcn.model import Model
from utils.experiment import Experiment
import torch.optim as optim
import torch.autograd
import torch
import numpy as np
from utils.loss_funcs import mpjpe_error
from utils.data_utils import define_actions
from utils.h36_3d_viz import visualize as vis


class H36_3DExperiment(Experiment):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: %s" % self.device)
        self.model_name = "h36_3d_best_model"
        self.skip_rate = 1
        self.joints_to_consider = 22
        self.model = Model(
            cfg["input_dim"],
            cfg["input_n"],
            cfg["output_n"],
            cfg["st_gcnn_dropout"],
            self.joints_to_consider,
            cfg["n_tcnn_layers"],
            cfg["tcnn_kernel_size"],
            cfg["tcnn_dropout"],
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
        dataset = datasets.Datasets(
            data_dir=self.cfg["data_dir"],
            input_n=self.cfg["input_n"],
            output_n=self.cfg["output_n"],
            skip_rate=self.skip_rate,
            device=self.device,
            split=0,
        )
        print(">>> Training dataset length: {:d}".format(dataset.__len__()))
        data_loader = DataLoader(
            dataset, batch_size=self.cfg["batch_size"], shuffle=True, num_workers=0, pin_memory=True
        )

        vald_dataset = datasets.Datasets(
            data_dir=self.cfg["data_dir"],
            input_n=self.cfg["input_n"],
            output_n=self.cfg["output_n"],
            skip_rate=self.skip_rate,
            device=self.device,
            split=1,
        )
        print(">>> Validation dataset length: {:d}".format(vald_dataset.__len__()))
        vald_loader = DataLoader(
            vald_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        dim_used = np.array(
            [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23,
             24, 25, 26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40,
             41, 42, 43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57,
             58, 59, 63, 64, 65, 66, 67, 68, 75, 76, 77, 78, 79, 80,
             81, 82, 83, 87, 88, 89, 90, 91, 92]
        )

        best_val_loss = np.inf
        for epoch in range(self.cfg["n_epochs"]):
            running_loss = 0
            n = 0
            self.model.train()
            for cnt, batch in enumerate(data_loader):
                batch = batch.to(self.device)
                batch_dim = batch.shape[0]
                n += batch_dim

                sequences_train = (
                    batch[:, 0 : self.cfg["input_n"], dim_used]
                    .view(-1, self.cfg["input_n"], len(dim_used) // 3, 3)
                    .permute(0, 3, 1, 2)
                )
                sequences_gt = batch[
                    :, self.cfg["input_n"] : self.cfg["input_n"] + self.cfg["output_n"], dim_used
                ].view(-1, self.cfg["output_n"], len(dim_used) // 3, 3)

                optimizer.zero_grad()

                sequences_predict = self.model(sequences_train).permute(0, 1, 3, 2)

                loss = mpjpe_error(sequences_predict, sequences_gt)

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
                for cnt, batch in enumerate(vald_loader):
                    batch = batch.to(self.device)
                    batch_dim = batch.shape[0]
                    n += batch_dim

                    sequences_train = (
                        batch[:, 0 : self.cfg["input_n"], dim_used]
                        .view(-1, self.cfg["input_n"], len(dim_used) // 3, 3)
                        .permute(0, 3, 1, 2)
                    )
                    sequences_gt = batch[
                        :,
                        self.cfg["input_n"] : self.cfg["input_n"] + self.cfg["output_n"],
                        dim_used,
                    ].view(-1, self.cfg["output_n"], len(dim_used) // 3, 3)

                    sequences_predict = self.model(sequences_train).permute(0, 1, 3, 2)

                    loss = mpjpe_error(sequences_predict, sequences_gt)
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
            torch.load(os.path.join(self.cfg["checkpoints_dir"], self.model_name))
        )
        self.model.eval()
        accum_loss = 0
        n_batches = 0  # number of batches for all the sequences
        actions = define_actions(self.cfg["actions_to_consider"])
        dim_used = np.array(
            [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22,
             23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 36, 37, 38,
             39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 52, 53, 54,
             55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68, 75, 76,
             77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92]
        )
        # joints at same loc
        joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
        index_to_ignore = np.concatenate(
            (joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2)
        )
        joint_equal = np.array([13, 19, 22, 13, 27, 30])
        index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

        for action in actions:
            running_loss = 0
            n = 0
            dataset_test = datasets.Datasets(
                data_dir=self.cfg["data_dir"],
                input_n=self.cfg["input_n"],
                output_n=self.cfg["output_n"],
                skip_rate=self.skip_rate,
                device=self.device,
                split=2,
                actions=[action],
            )
            print(">>> test action for sequences: {:d}".format(dataset_test.__len__()))

            test_loader = DataLoader(
                dataset_test,
                batch_size=self.cfg["batch_size_test"],
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            for cnt, batch in enumerate(test_loader):
                with torch.no_grad():
                    batch = batch.to(self.device)
                    batch_dim = batch.shape[0]
                    n += batch_dim

                    all_joints_seq = batch.clone()[
                        :, self.cfg["input_n"] : self.cfg["input_nse"] + self.cfg["output_n"], :
                    ]

                    sequences_train = (
                        batch[:, 0 : self.cfg["input_n"], dim_used]
                        .view(-1, self.cfg["input_n"], len(dim_used) // 3, 3)
                        .permute(0, 3, 1, 2)
                    )
                    sequences_gt = batch[
                        :, self.cfg["input_n"] : self.cfg["input_n"] + self.cfg["output_n"], :
                    ]

                    sequences_predict = (
                        self.model(sequences_train)
                        .permute(0, 1, 3, 2)
                        .contiguous()
                        .view(-1, self.cfg["output_n"], len(dim_used))
                    )

                    all_joints_seq[:, :, dim_used] = sequences_predict

                    all_joints_seq[:, :, index_to_ignore] = all_joints_seq[:, :, index_to_equal]

                    loss = mpjpe_error(
                        all_joints_seq.view(-1, self.cfg["output_n"], 32, 3),
                        sequences_gt.view(-1, self.cfg["output_n"], 32, 3),
                    )
                    running_loss += loss * batch_dim
                    accum_loss += loss * batch_dim

            print(
                "loss at test subject for action : " + str(action) + " is: " + str(running_loss / n)
            )
            n_batches += n
        print("overall average loss in mm is: " + str(accum_loss / n_batches))

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
            self.cfg["actions_to_consider"],
        )
