import argparse
import os
import torch
import numpy as np
import time
import shutil
from utils.metrics import metric
from exp.exp_model import Exp_Model

parser = argparse.ArgumentParser(description='SimTFV')

parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')

parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--long_input_len', type=int, default=96, help='input length')
parser.add_argument('--short_input_len', type=int, default=96, help='input length')
parser.add_argument('--pred_len', type=str, default='192,336,720', help='prediction length')

parser.add_argument('--enc_in', type=int, default=7, help='input variable number')
parser.add_argument('--dec_out', type=int, default=7, help='output variable number')
parser.add_argument('--d_model', type=int, default=32, help='hidden dims of model')
parser.add_argument('--encoder_layers', type=str, default='2,2,4', help='num of layers in each encoder stage')
parser.add_argument('--decoder_layers', type=str, default='1,1,2', help='num of layers in each decoder stage')
parser.add_argument('--patch_size', type=int, default=6,
                    help='the initial patch size in patch-wise attention')
parser.add_argument('--Not_use_CV', action='store_true',
                    help='whether not to adopt the cross-variable attention in TVA'
                    , default=False)
parser.add_argument('--decoder_IN', action='store_true',
                    help='whether to use decoder_IN'
                    , default=False)

parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=5, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--decay', type=float, default=0.5, help='decay rate of learning rate per epoch')
parser.add_argument('--patience', type=int, default=1, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--save_loss', action='store_true', help='whether saving results and checkpoints', default=False)
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_gpu', action='store_true',
                    help='whether to use gpu, it is automatically set to true if gpu is available in your device'
                    , default=False)
parser.add_argument('--train', action='store_true',
                    help='whether to train'
                    , default=False)
parser.add_argument('--reproducible', action='store_true',
                    help='whether to make results reproducible'
                    , default=False)

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() else False

if args.reproducible:
    np.random.seed(4321)  # reproducible
    torch.manual_seed(4321)
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.deterministic = False

args.pred_len = [int(predl) for predl in args.pred_len.replace(' ', '').split(',')]
args.encoder_layers = [int(el) for el in args.encoder_layers.replace(' ', '').split(',')]
args.decoder_layers = [int(dl) for dl in args.decoder_layers.replace(' ', '').split(',')]

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

data_parser = {
    'ECL': {'data': 'ECL.csv', 'root_path': './data/ECL/', 'M': [321, 321]},
    'Solar': {'data': 'solar_AL.csv', 'root_path': './data/Solar/', 'M': [137, 137]},
    'Wind': {'data': 'Wind.csv', 'root_path': './data/Wind/', 'M': [28, 28]},
    'Hydro': {'data': 'Hydro_BXX.csv', 'root_path': './data/Hydro_BXX/', 'M': [14, 14]},
}

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.root_path = data_info['root_path']
    args.enc_in, args.dec_out = data_info['M']

lr = args.learning_rate
print('Args in experiment:')
print(args)

Exp = Exp_Model
for ii in range(args.itr):
    if args.train:
        setting = '{}_ll{}_pl{}_{}'.format(args.data, args.long_input_len,
                                           args.pred_len, ii)
        print('>>>>>>>start training| pred_len:{}, settings: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.
              format(args.pred_len, setting))
        try:
            exp = Exp(args)  # set experiments
            exp.train(setting)
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from forecasting early')

        print('>>>>>>>testing| pred_len:{}: {}<<<<<<<<<<<<<<<<<'.format(args.pred_len, setting))
        exp = Exp(args)  # set experiments
        exp.test(setting, load=True)
        torch.cuda.empty_cache()
        args.learning_rate = lr
    else:
        setting = '{}_ll{}_pl{}_{}'.format(args.data, args.long_input_len,
                                           args.pred_len, ii)
        print('>>>>>>>testing| pred_len:{} : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.pred_len, setting))
        exp = Exp(args)  # set experiments

        exp.test(setting, load=True)
        torch.cuda.empty_cache()
        args.learning_rate = lr

path1 = './result.csv'
if not os.path.exists(path1):
    with open(path1, "a") as f:
        write_csv = ['Time', 'Data', 'input_len', 'pred_len', 'MODWT', 'MSE',
                     'MAE']
        np.savetxt(f, np.array(write_csv).reshape(1, -1), fmt='%s', delimiter=',')
        f.flush()
        f.close()

print('>>>>>>>writing results<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
first_setting = '{}_ll{}_pl{}_{}'.format(args.data, args.long_input_len, args.pred_len, 0)
first_folder_path = './results/' + first_setting
num_of_files = len([f for f in os.listdir(first_folder_path) if os.path.isfile(os.path.join(first_folder_path, f))])
num_of_test = num_of_files // 2
print('test windows number: ' + str(num_of_test))

for predl in args.pred_len:
    mses = []
    maes = []
    for i in range(num_of_test):
        pred_total = 0
        true = None
        for ii in range(args.itr):
            setting = '{}_ll{}_pl{}_{}'.format(args.data, args.long_input_len, args.pred_len, ii)
            folder_path = './results/' + setting + '/'
            pred_path = folder_path + 'pred_{}.npy'.format(i)
            pred = np.load(pred_path)
            pred_total += pred
            if true is None:
                true_path = folder_path + 'true_{}.npy'.format(i)
                true = np.load(true_path)
        pred = pred_total / args.itr
        mae, mse = metric(pred[:, :predl, :], true[:, :predl, :])
        mses.append(mse)
        maes.append(mae)

    mse = np.mean(mses)
    mae = np.mean(maes)
    print('|Mean|mse:{}, mae:{}'.format(mse, mae))
    path = './result.log'
    with open(path, "a") as f:
        f.write('|{}|input_len{}_pred_len{}: '.format(
            args.data, args.long_input_len, predl) + '\n')
        f.write('mse:{}, mae:{}'.
                format(mse, mae) + '\n')
        f.flush()
        f.close()

    with open(path1, "a") as f:
        f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        f.write(',{},{},{},{},{}'.
                format(args.data, args.long_input_len, predl
                       , mse, mae) + '\n')
        f.flush()
        f.close()

if not args.save_loss:
    for ii in range(args.itr):
        setting = '{}_ll{}_pl{}_{}'.format(args.data, args.long_input_len, args.pred_len, ii)
        dir_path = os.path.join(args.checkpoints, setting)
        check_path = dir_path + '/' + 'checkpoint.pth'
        if os.path.exists(check_path):
            os.remove(check_path)
            os.removedirs(dir_path)

        folder_path = './results/' + setting
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
