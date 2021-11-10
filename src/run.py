from utils.config import read_config
from stsgcn.experiment_amass_3d import Amass3DExperiment
from stsgcn.experiment_h36_3d import H36_3DExperiment
from stsgcn.experiment_h36_ang import H36AngExperiment

cfg = read_config("configs/stsgcn_config.yaml")

if cfg["experiment"] == "amass_3d":
    experiment = Amass3DExperiment(cfg)
elif cfg["experiment"] == "h36_3d":
    experiment = H36_3DExperiment(cfg)
elif cfg["experiment"] == "h36_ang":
    experiment = H36AngExperiment(cfg)
else:
    raise NotImplementedError()

if cfg["mode"] == "train":
    experiment.train()
elif cfg["mode"] == "test":
    experiment.test()
elif cfg["mode"] == "visualize":
    experiment.visualize()
else:
    raise NotImplementedError()
