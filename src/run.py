from stsgcn.train import train
import argparse

parser = argparse.ArgumentParser(description='Arguments for running the scripts')
parser.add_argument('--data_dir', type=str, default='/cluster/scratch/aarslan/dlproject_datasets/', help='path to the unzipped dataset directories(H36m/AMASS/3DPW)')
parser.add_argument('--gen_model', type=str, default='stsgcn', help='generator model')
parser.add_argument('--dataset', type=str, default='amass_3d', help='dataset to run models on')
parser.add_argument('--output_n', type=int, default=10, help="number of model's output frames")
parser.add_argument('--gen_clip_grad', type=float, default=10.0, help='select max norm to clip gradients')
parser.add_argument('--recurrent_cell', type=str, default='gru', help='recurrent cell type')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--gen_lr', type=int, default=0.01, help='generator learning rate')
parser.add_argument('--early_stop_patience', type=int, default=10, help='early stop patience')
parser.add_argument('--use_disc', dest='feature', action='store_true')

args = parser.parse_args()

train("../configs/config.yaml", args)
