import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle
from datetime import datetime

from symbiotic_learning.classify.readout import Readout
import symbiotic_learning.classify.utils as classify
import symbiotic_learning.loss as loss

from model import Model

def main():
    torch.cuda.empty_cache()

    now = datetime.now()
    date_and_time = now.strftime('%d_%m_%y_%H_%M_%S')
    
    parser = argparse.ArgumentParser(prog='main',
                    description='main.py: Robot Anomaly Classification with Symbiotic Learning.',
                    epilog='Full description TBD.')

    parser.add_argument('-n', '--num_preR', default=2, type=int)
    parser.add_argument('-c', '--collab_params', nargs='*', type=float, help='Collaboration parameters. Specify <num> values in the range (0,1).')
    
    parser.add_argument('--comment', default='', type=str)

    parser.add_argument('--seq_len', default=8, type=int)
    parser.add_argument('--kernel_size', default=2, type=int)
    parser.add_argument('--conv_channels', default=2, type=int)
    parser.add_argument('--fc_channels', default=2, type=int)

    parser.add_argument('--readout_hidden_dim', default=2, type=int)
    parser.add_argument('--readout_num_hidden', default=1, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--attn_dropout', default=0.0, type=float)
    parser.add_argument('--num_fc', default=1, type=int)
    parser.add_argument('--temp', default=1.0, type=float)
    parser.add_argument('--lamb', default=1.0, type=float, help='Responsibility parameter lambda. Enables blame loss term after uplift.')
    
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--uplift', default=5, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    
    args = parser.parse_args()  
    num_preR = args.num_preR
    collab_params = args.collab_params

    comment = args.comment
    
    input_dim = args.seq_len
    kernel_size = args.kernel_size
    conv_channels = args.conv_channels
    fc_channels = args.fc_channels

    readout_hidden_dim = args.readout_hidden_dim
    readout_num_hidden = args.readout_num_hidden
    num_heads = args.num_heads
    attn_dropout = args.attn_dropout
    num_fc = args.num_fc
    temp = args.temp
    lamb = args.lamb
    
    epochs = args.epochs
    uplift = args.uplift
    batch_size = args.batch_size
    
    df_names = ['train', 'valid', 'test']
    
    criterion = torch.nn.CrossEntropyLoss()
    pct_start = (uplift/2) / epochs
    
    if comment != '':
        save_path = f'./sym_logs/{date_and_time}_{comment}'
    else:
        save_path = f'./sym_logs/{date_and_time}'

    os.mkdir(save_path)
    
    models = []
    opts = []
    scheds = []

    data_loaders = []
    for name in df_names:
        with open(f'./{name}_Xs.pkl', 'rb') as file:
            Xs = pickle.load(file)
        with open(f'./{name}_Ys.pkl', 'rb') as file:
            Ys = pickle.load(file)

        tensor_X = torch.from_numpy(np.concatenate(Xs)).permute(0,2,1)
        tensor_Y = torch.from_numpy(np.concatenate(Ys))
        tensor_data = TensorDataset(tensor_X, tensor_Y)
        data_loaders.append(DataLoader(tensor_data, batch_size=batch_size, shuffle=True))
    
    num_train_batches = len(data_loaders[0])
    
    for _ in range(num_preR):
        model_i = Model(input_dim, kernel_size, in_channels=9, conv_channels=conv_channels, num_fc=num_fc, fc_channels=fc_channels, stride=1, padding='same', dilation=1, padding_mode='zeros', output_dim=6, num_preR=num_preR)
        model_i = model_i.cuda()
        model_i.double()
    
        opt_i = optim.AdamW(model_i.parameters(), lr=1e-7, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        sched_i = torch.optim.lr_scheduler.OneCycleLR(opt_i, max_lr=5e-5, steps_per_epoch=num_train_batches, epochs=epochs, pct_start=pct_start) 
    
        models.append(model_i)
        opts.append(opt_i)
        scheds.append(sched_i)
    
    readout = Readout(hidden_dim=readout_hidden_dim, num_hidden=readout_num_hidden, num_classes=6, num_heads=num_heads, num_preR=num_preR, preR_dim=fc_channels, attn_dropout=attn_dropout)
    readout.double()
    
    models.append(readout)
    opt_F = optim.AdamW(models[-1].parameters(), lr=1e-7, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    opts.append(opt_F)
    
    sched_F = torch.optim.lr_scheduler.OneCycleLR(opts[-1], max_lr=5e-5, steps_per_epoch=num_train_batches, epochs=epochs-uplift+1)
    scheds.append(sched_F)

    num_trainable_params = sum(p.numel() for p in models[0].parameters() if p.requires_grad)
    num_trainable_params_readout = sum(p.numel() for p in readout.parameters() if p.requires_grad)

    print()
    print('Total Number of Trainable Parameters: ', num_preR*num_trainable_params+num_trainable_params_readout)
    print('\t-Total pre-Readout Trainable Parameters: ', num_preR*num_trainable_params)
    print('\t\t-Number of Trainable Parameters per pre-Readout Model: ', num_trainable_params)
    print('\t-Number of Trainable Parameters in Readout Block: ', num_trainable_params_readout)
    print()

    with open(f"{save_path}/param_breakdown.txt", "w", encoding="utf-8") as file:
        file.write(f"Total Number of Trainable Parameters: {num_preR*num_trainable_params+num_trainable_params_readout}\n")
        file.write(f"\tTotal pre-Readout Trainable Parameters: {num_preR*num_trainable_params}\n")
        file.write(f"\t\t-Number of Trainable Parameters per pre-Readout Model: {num_trainable_params}\n")
        file.write(f"\tNumber of Trainable Parameters in Readout Block: {num_trainable_params_readout}")
        
    classify.train(epochs, models, opts, scheds, data_loaders, collab_params, temp, criterion, uplift=uplift, eps=1e-7, lamb=lamb, save_path=save_path, save_best=True)

if __name__ == '__main__':
    main()