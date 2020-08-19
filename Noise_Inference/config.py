
import argparse


parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def str2bool(v):
    return v.lower() in ('true', '1')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--batch_size', type=int, default= 128, help='batch size')
data_arg.add_argument('--nCells', type=int, default=10, help='number of cells')
data_arg.add_argument('--nMuts', type=int, default=7, help='number of mutations')
data_arg.add_argument('--output_dir', type=str,  help='output directory')
data_arg.add_argument('--h5_dir', type=str,  help='Data in h5 format directory')
data_arg.add_argument('--ms_dir', type=str,  help='MS program directory')
data_arg.add_argument('--nTrain', type=int, default=10000, help='number of train instances')
data_arg.add_argument('--nTest', type=int, default=100, help='number of test instances')
data_arg.add_argument('--alpha', type=float, default=0.0001, help='False positive rate')
data_arg.add_argument('--beta', type=float, default=0.02, help='False negative rate')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--nb_epoch', type=int, default=1500, help='nb epoch')
train_arg.add_argument('--lexSort', type=str2bool, default=True, help='sort the column of the matrices lexicographically') 

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_config():
    config, _ = get_config()
    print('\n')
    print('Data Config:')
    print('* Number of Cells:', config.nCells)
    print('* Number of Mutations:', config.nMuts)
    print('* Lexsorted:', config.lexSort)
    print('* Number of training instances:', 2*config.nTrain)
    print('* Number of test instances:', 2*config.nTest)
    print('\n')


