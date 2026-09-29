import kagglehub
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pickle
import numpy.random as random

def minmax_norm(col, x_min=False, x_max=False):

    if x_min==False and x_max==False:
        x_min = np.min(col)
        x_max = np.max(col)
    elif np.logical_xor(x_min, x_max):
        raise Exception('Must provide both x_min and x_max or neither of them.')
    
    normed = (col-x_min)/(x_max-x_min)

    return (x_min, x_max), normed

def main():

    ## Adapted from https://github.com/hkayann/Real-time-Anomaly-Detection-in-Industrial-Robotic-Arms-via-TinyML/blob/main/notebooks/main.ipynb ##

    parser = argparse.ArgumentParser(prog='process_data',
                    description='process_data.py: Process IMU datasets and save relevant numpy arrays.',
                    epilog='Full description TBD.')
    parser.add_argument('--seq_len', default=32, type=int, help='Number of frames to stack as one sequence.')
    parser.add_argument('--max_samples', default=100000, type=int)
    parser.add_argument('--seed', default=717, type=int)

    args = parser.parse_args()  
    seq_len = args.seq_len
    max_samples = args.max_samples
    seed = args.seed

    random.seed(seed=seed)
    
    drops = ['name', 'time']

    dataset_IMU = pd.read_csv('./data/industrial-robotic-arm-imu-data-casper-1-and-2/versions/2/IMU_10Hz.csv')

    dataset_IMU.drop(columns=['name', 'time'], inplace=True)
    dataset_IMU = dataset_IMU.iloc[120:]
    dataset_IMU.reset_index(drop=True, inplace=True)
    labels = np.full(len(dataset_IMU),0)
    dataset_IMU['label'] = labels

    load_path = './data/industrial-robotic-arm-imu-data-casper-1-and-2/versions/2'

    df_hitting_arm = pd.read_csv(f'{load_path}/IMU_hitting_arm.csv', header=None).drop(columns=[0, 1]).iloc[1:].reset_index(drop=True)
    df_hitting_platform = pd.read_csv(f'{load_path}/IMU_hitting_platform.csv', header=None).drop(columns=[0, 1]).iloc[1:].reset_index(drop=True)
    df_earthquake = pd.read_csv(f'{load_path}/IMU_earthquake.csv', header=None).drop(columns=[0, 1]).iloc[1:].reset_index(drop=True)
    df_extra_weight = pd.read_csv(f'{load_path}/IMU_extra_weigth.csv', header=None).drop(columns=[0, 1]).iloc[1:].reset_index(drop=True)
    df_magnet = pd.read_csv(f'{load_path}/IMU_magnet.csv', header=None).drop(columns=[0, 1]).iloc[1:].reset_index(drop=True)
    
    df_hitting_arm.columns = range(df_hitting_arm.shape[1])
    df_hitting_platform.columns = range(df_hitting_platform.shape[1])
    df_earthquake.columns = range(df_earthquake.shape[1])
    df_extra_weight.columns = range(df_extra_weight.shape[1])
    df_magnet.columns = range(df_magnet.shape[1])
    
    df_hitting_arm = df_hitting_arm.apply(pd.to_numeric, errors='coerce')
    df_hitting_platform = df_hitting_platform.apply(pd.to_numeric, errors='coerce')
    df_earthquake = df_earthquake.apply(pd.to_numeric, errors='coerce')
    df_extra_weight = df_extra_weight.apply(pd.to_numeric, errors='coerce')
    df_magnet = df_magnet.apply(pd.to_numeric, errors='coerce')
    
    print(f"Hitting Arm Length: {len(df_hitting_arm)}")
    print(f"Hitting Platform Length: {len(df_hitting_platform)}")
    print(f"Earthquake Length: {len(df_earthquake)}")
    print(f"Extra Weigth Length: {len(df_extra_weight)}")
    print(f"Magnet Length: {len(df_magnet)}")
    
    # Define the feature names
    feature_names = ["accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ", "magX", "magY", "magZ"]
    
    # Assign feature names to columns for each DataFrame
    df_hitting_arm.columns = feature_names
    df_hitting_platform.columns = feature_names
    df_earthquake.columns = feature_names
    df_extra_weight.columns = feature_names
    df_magnet.columns = feature_names
    
    
    for i, df in enumerate([df_hitting_arm, df_hitting_platform, df_earthquake, df_extra_weight, df_magnet]):
        labels = np.full(len(df),i+1)
        df['label'] = labels

    train_frac = 0.5
    val_frac = 0.25
    test_frac = 0.25
 
    rand_num = random.rand(1)[0]
    print('Random Number: ', rand_num)
    nominal_offset = int(rand_num*len(dataset_IMU))
    print('Nominal Offset: ', nominal_offset)

    
    train_df = []
    val_df = []
    test_df = []
    
    datasets = [dataset_IMU, df_hitting_arm, df_hitting_platform, df_earthquake, df_extra_weight, df_magnet]
    
    for i, df in enumerate(datasets):

        num_train_samples = min( [int(train_frac*len(df)), max_samples] )
        num_val_samples = min( [int(val_frac*len(df)), max_samples] )
        num_test_samples = min( [int(test_frac*len(df)), max_samples] )

        max_idx = nominal_offset+num_train_samples+num_val_samples+num_test_samples
        
        if i == 0:

            if max_idx > len(dataset_IMU):
                train_offset = (nominal_offset+max_idx)//len(dataset_IMU)
            else:
                train_offset = nominal_offset
                
            train_df.append( df.iloc[train_offset:num_train_samples+train_offset] )
            val_df.append( df.iloc[num_train_samples+train_offset:num_train_samples+num_val_samples+train_offset] )
            test_df.append( df.iloc[num_train_samples+num_val_samples+train_offset:num_train_samples+num_val_samples+num_test_samples+train_offset] )
            
        else:
            train_df.append( df.iloc[:num_train_samples] )
            val_df.append( df.iloc[num_train_samples:num_train_samples+num_val_samples] )
            test_df.append( df.iloc[num_train_samples+num_val_samples:] )
        
    mins = []
    maxs = []
    
    train_df = pd.concat(train_df)
    val_df = pd.concat(val_df)
    test_df = pd.concat(test_df)
    
    for feature in feature_names:
        extrema, train_normed = minmax_norm(train_df[feature])
    
        x_min, x_max = extrema
        mins.append(x_min)
        maxs.append(x_max)
        
        train_df[feature] = train_normed
    
        _, val_normed = minmax_norm(val_df[feature], x_min=x_min, x_max=x_max)
        val_df[feature] = val_normed
    
        _, test_normed = minmax_norm(test_df[feature], x_min=x_min, x_max=x_max)
        test_df[feature] = test_normed
    
    data_loaders = []
    
    df_names = ['train', 'valid', 'test']
    
    for name, full_df in zip(df_names, [train_df, val_df, test_df]):
    
        Xs = []
        Ys = []
    
        for _ , df in full_df.groupby('label'):
        
            ts_data = timeseries_dataset_from_array(
                df.to_numpy()[:,:-1],
                df.to_numpy()[:,-1],
                seq_len,
                sequence_stride=1,
                sampling_rate=1,
                batch_size=64,
                shuffle=False,
                seed=seed,
                start_index=None,
            )
        
            ts_data = ts_data.unbatch()
            ts_data = list(ts_data.as_numpy_iterator())
            
            X = np.stack([x.astype(np.float64) for x, y in ts_data])
            Y = np.array([y.astype(np.int64) for x, y in ts_data])
            
            Xs.append(X)
            Ys.append(Y)

        with open(f'./{name}_Xs.pkl', 'wb') as file:
            pickle.dump(Xs, file)

        with open(f'./{name}_Ys.pkl', 'wb') as file:
            pickle.dump(Ys, file)

if __name__ == "__main__":
    
    main()