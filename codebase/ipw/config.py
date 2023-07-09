from __future__ import print_function
import argparse
import time

def str2bool(v):
    return v is True or v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

parser.add_argument('--name', type=str, default='Debias', help='The main model name')
parser.add_argument('--time', type=str, default='', help='Current time')
#Learning process
parser.add_argument('--epoch_max', type=int, default=16, help='The learning epoches')
parser.add_argument('--iter_save', type=int, default=30, help='The save turn')
parser.add_argument('--pt_load_path', type=str, default='')
parser.add_argument('--feature_data',  type=bool, default=False, help="If data have side information")
parser.add_argument('--experiment_id', type=int, default=0, help="The expriment id")

#Pretrain network
parser.add_argument('--train_mode', type=str, default='pretrain', choices=['pretrain', 'train', 'test', 'save_imputation'], help='Weighted learning')
parser.add_argument('--use_weight', type=str2bool, default=False, help='Weighted learning')

#dev network
parser.add_argument('--model_dir', type=str, default='', help='The model dir')

#Used to be an option, but now is solved
#pretrain_arg.add_argument('--pretrain_type',type=str,default='wasserstein',choices=['wasserstein','gan'])
parser.add_argument('--user_dim', type=int, default=1, help="User dimension (set 1 means using user and item id)")
parser.add_argument('--item_dim', type=int, default=1, help="Item dimension (set 1 means using user and item id)")
parser.add_argument('--user_size', type=int, default=1000, help="User size")
parser.add_argument('--item_size', type=int, default=1720, help="Item size")
parser.add_argument('--user_item_size', type=int, nargs='+', default=[1000, 1720], help="User and Item size")
parser.add_argument('--user_emb_dim', type=int, default=64, help="Item embedding dimension")
parser.add_argument('--item_emb_dim', type=int, default=64, help="Item embedding dimension")
parser.add_argument('--ctr_layer_dims',  type=int, nargs='+', default=[64, 32, 16],   help="Hidden layer dimension of ctr prediction model")
parser.add_argument('--debias_mode',  type=str, default='Propensitylearnt_Mode', choices=['Pretrain','Propensity_Mode','SNIPS_Mode','Propensitylearnt_Mode', 'SNIPSlearnt_Mode','Direct', 'DoublyRobust_Mode', 'Propensity_DR_Mode', 'Uniform_DR_Mode', 'CVIB', 'ATT', 'ACL'], help="The mode of weight")
parser.add_argument('--pretrain_mode',  type=str, default='propensity', choices=['propensity','imputation','uniform_imputation'], help="The mode of weight")
parser.add_argument('--impression_len', type=int, default=5, help="The impression length")


# new added
parser.add_argument('--weight_decay', type=float, default=0.001, help="weight_decay in BPR model")
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--ctr_classweight', type=int, nargs='+', default=[1, 1],
                    help='the class weight of objective function of prediction model')
parser.add_argument('--is_debias', type=str2bool, default=False, help='Using feature balance as representation')
#IPM function

parser.add_argument('--IMP_mode',  type=str, default='gaussian', choices=['gaussian', 'functional'], help="The mode of IPM function")
parser.add_argument('--kernel_mul', type=float, default=2.0, help='The mean of gaussian kernel')
parser.add_argument('--kernel_num', type=int, default=5, help='The number of kernel number')
parser.add_argument('--fix_sigma', type=float, default=None, help='The variance of gaussian kernel')

parser.add_argument('--ipm_layer_dims',  type=int, nargs='+', default=[64, 32, 16], help='Hidden layer dimension of IPM prediction model')
parser.add_argument('--ipm_embedding',  type=int, default=16, help='Hidden layer dimension of IPM prediction model')

# downstream task
parser.add_argument('--downstream', type=str, default='MLP',
                    choices=['MLP', 'gmfBPR', 'bprBPR', 'mlpBPR', 'NeuBPR', 'LightGCN', 'DCN'], help="The mode of weight")

#density ratio function
parser.add_argument('--dr_layer_dim',  type=int, nargs='+', default=[32, 32, 16], help='Hidden layer dimension of IPM prediction model')
parser.add_argument('--dr_step',  type=int, default=60, help='Load n-th density ratio model')
#data
parser.add_argument('--dataset', type=str, default='pcic')

parser.add_argument('--batch_size', type=int, default=8192)
parser.add_argument('--num_worker', type=int, default=8, help='number of threads to use for loading and preprocessing data')


def get_config():
    config, unparsed = parser.parse_known_args()
    current_time = time.localtime(time.time())
    config.time = '{}_{}_{}_{}'.format(current_time.tm_mon, current_time.tm_mday, current_time.tm_hour, current_time.tm_min)
    model_name = [
        ('name={:s}',  config.name),
        ('use_weight={}',  config.use_weight),
        ('debias_mode={}',  config.debias_mode)
    ]
    config.model_dir = '_'.join([t.format(v) for (t, v) in model_name])
    print('Loaded ./config.py')
    return config, unparsed

if __name__=='__main__':
    #for debug of config
    config, unparsed = get_config()

'''
# usage
from causal_controller.config import get_config as get_cc_config
cc_config,_=get_cc_config()
'''