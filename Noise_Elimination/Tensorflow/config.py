
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--hidden_dim', type=int, default=128, help='actor LSTM num_neurons')
net_arg.add_argument('--num_heads', type=int, default=16, help='actor input embedding')  ###
net_arg.add_argument('--num_stacks', type=int, default=3, help='actor LSTM num_neurons')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--batch_size', type=int, default= 128, help='batch size')
data_arg.add_argument('--input_dimension', type=int, default=4, help='city dimension')
data_arg.add_argument('--nCells', type=int, default=10, help='number of cells')
data_arg.add_argument('--nMuts', type=int, default=10, help='number of mutations')
data_arg.add_argument('--output_dir', type=str,  help='output matrices directory')
data_arg.add_argument('--nTestMats', type=int, default=100, help='number of test instances')
data_arg.add_argument('--alpha', type=float, default=0.0001, help='False positive rate')
data_arg.add_argument('--beta', type=float, default=0.02, help='False negative rate')
data_arg.add_argument('--gamma', type=float, default=0.04, help='hyperparameter in the cost function')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--nb_epoch', type=int, default=100, help='nb epoch')
train_arg.add_argument('--lr1_start', type=float, default=0.001, help='actor learning rate')
train_arg.add_argument('--lr1_decay_step', type=int, default=5000, help='lr1 decay step')
train_arg.add_argument('--lr1_decay_rate', type=float, default=0.96, help='lr1 decay rate')

train_arg.add_argument('--ma', type=float, default=0.99, help='update factor moving average baseline')
train_arg.add_argument('--init_baseline', type=float, default=7.0, help='initial baseline - REINFORCE')

train_arg.add_argument('--temperature', type=float, default=3.0, help='pointer_net initial temperature')
train_arg.add_argument('--C', type=float, default=10.0, help='pointer_net tan clipping')

# Misc
misc_arg = add_argument_group('User options') #####################################################

misc_arg.add_argument('--inference_mode', type=str2bool, default=True, help='switch to inference mode when model is trained') 
# misc_arg.add_argument('--restore_mod_fromel', type=str2bool, default=True, help='whether or not model is retrieved')

misc_arg.add_argument('--save_to', type=str, default='20/model', help='saver sub directory')
misc_arg.add_argument('--restore_from', type=str, default='20/model', help='loader sub directory')  ###
misc_arg.add_argument('--log_dir', type=str, default='summary/20/repo', help='summary writer log directory') 
misc_arg.add_argument('--ms_dir', type=str, default='/u/ms', help='ms program directory') 

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_config():
    config, _ = get_config()
    print('\n')
    print('Data Config:')
    print('* Batch size:',config.batch_size)
    print('* Sequence length:',config.nCells*config.nMuts)
    print('\n')
    print('Network Config:')
    print('* Actor hidden_dim:',config.hidden_dim)
    print('\n')
    if config.inference_mode==False:
        print('Training Config:')
        print('* Nb epoch:',config.nb_epoch)
    else:
        print('Testing Config:')
        print('Number of test matrices:', config.nTestMats)
    print('\n')