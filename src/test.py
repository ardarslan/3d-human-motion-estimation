from stsgcn.test import test
import argparse

parser = argparse.ArgumentParser(description='Arguments for running the scripts')
parser.add_argument('--data_dir',        type=str, default='../dlproject_datasets/', help='path to the unzipped dataset directories(H36m/AMASS/3DPW)')
parser.add_argument('--dataset',         type=str, default='amass_3d', help='dataset to run models on')
parser.add_argument('--output_n',        type=int, default=25, help="number of model's output frames")

parser.add_argument('--gen_model',       type=str, default='stsgcn', help='generator model')
parser.add_argument('--recurrent_cell',  type=str, default='gru', help='recurrent cell type')
parser.add_argument('--use_disc',        dest='use_disc', action='store_true')

parser.add_argument('--batch_size',      type=int, default=256, help='batch size')
parser.add_argument('--gen_lr',              type=float, default=0.01, help='generator learning rate')
parser.add_argument('--gen_clip_grad',       type=float, default=None, help='select max norm to clip gradients')
parser.add_argument('--gen_gamma',           type=float, default=0.1, help='generator gamma for learning rate scheduling')
parser.add_argument('--gen_milestones',      type=int, nargs='*', default=[15, 25, 35, 40])
parser.add_argument('--early_stop_patience', type=int, default=10, help='early stop patience')


parser.add_argument('--model_loc',           type=str, default='../logs/200800123', help='location containing best_model after training')
args = parser.parse_args()

test("../configs/test_config.yaml", args)
