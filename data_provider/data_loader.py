import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single
import json
from utils.run_metrics import user_definable_IOH

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:] # (total_length, features)
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]] # (total_length, 1)

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end] # (seq_len, features)
        seq_y = self.data_y[r_begin:r_end] # (label_len+pred_len, features)
        seq_x_mark = self.data_stamp[s_begin:s_end] # (seq_len, 4)
        seq_y_mark = self.data_stamp[r_begin:r_end] # (label_len+pred_len, 4)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           max(0, cut_point - self.label_len):min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from ts files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .ts files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern='*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)

class VitalDBLoader(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='Solar8000/ART_MBP', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, sample_step=5):
        # size [seq_len, label_len, pred_len]
        self.args = args

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.sample_step = sample_step

        self.root_path = root_path
        self.data_path = data_path
        
        # 定义时间序列特征 - 顺序固定，不可更改
        self.time_series_features = [
            'Solar8000/BT',    # 体温 (Body Temperature) - 特征索引: 0
            'Solar8000/HR',    # 心率 (Heart Rate) - 特征索引: 1
            'Solar8000/ART_DBP', # 动脉舒张压 (Arterial Diastolic Blood Pressure) - 特征索引: 2
            'Solar8000/ART_MBP',  # 动脉平均血压 (Arterial Mean Blood Pressure) - 特征索引: 3, 目标变量
        ]
        

        # 数据存储结构
        self.samples = []  # 格式：(caseid, start_idx)
        self.case_data = {}  # {caseid: full_data}
        
        self.__read_data__()

    def __read_data__(self):
        """读取数据并进行预处理"""
        # 根据flag选择对应的jsonl文件
        file_map = {
            0: 'train.jsonl',
            1: 'val.jsonl',
            2: 'test.jsonl'
        }
        file_path = os.path.join(self.root_path, file_map[self.set_type])
        
        # 读取数据
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    caseid = record['caseid']
                    case_data = {}
                    for feature in self.time_series_features:
                        if feature in record:
                            case_data[feature] = record[feature]
                    if case_data:  # 确保至少有一个特征
                        self.case_data[caseid] = case_data
                except:
                    continue

        # 预生成所有有效样本
        self.__precompute_samples__()
        
        # 如果是训练集，进行数据增强
        if self.flag == 'train' and self.args.augmentation_ratio > 0:
            self.__augment_data__()
            
        # 为每个特征创建一个scaler
        self.scalers = {feature: StandardScaler() for feature in self.time_series_features}
        
        # 归一化处理
        if self.scale:
            # 如果是训练集，拟合scaler
            if self.set_type == 0:  # train
                for feature in self.time_series_features:
                    train_data = []
                    for caseid, data in self.case_data.items():
                        if feature in data:
                            train_data.extend(data[feature])
                    train_data = np.array(train_data).reshape(-1, 1)
                    self.scalers[feature].fit(train_data)
            else:
                # 对于验证集和测试集，需要从训练集获取scaler
                train_file_path = os.path.join(self.root_path, 'train.jsonl')
                with open(train_file_path, 'r') as f:
                    train_data = {feature: [] for feature in self.time_series_features}
                    for line in f:
                        try:
                            record = json.loads(line)
                            for feature in self.time_series_features:
                                if feature in record:
                                    train_data[feature].extend(record[feature])
                        except:
                            continue
                
                # 拟合scaler
                for feature in self.time_series_features:
                    if train_data[feature]:
                        train_data_array = np.array(train_data[feature]).reshape(-1, 1)
                        self.scalers[feature].fit(train_data_array)
            
            # 对所有数据进行归一化
            for caseid, data in self.case_data.items():
                for feature in self.time_series_features:
                    if feature in data:
                        data[feature] = self.scalers[feature].transform(
                            np.array(data[feature]).reshape(-1, 1)
                        ).flatten()

    def __precompute_samples__(self):
        """预生成所有有效样本"""
        required_length = self.seq_len + self.pred_len

        for caseid, data in self.case_data.items():
            max_start = len(data['Solar8000/ART_MBP']) - required_length

            if max_start < 0:
                continue
                
            # 生成该病例段内的所有样本起始位置
            for sample_start in range(0, max_start + 1, self.sample_step):
                # 检查数据是否有效（无突变）
                segment_data = data['Solar8000/ART_MBP'][sample_start:sample_start+required_length]
                if (np.abs(np.diff(segment_data)) > 60).any():  
                    continue # abrupt change -> noise

                self.samples.append((caseid, sample_start))

    def _find_ioh_segments(self, x_future):
        """找出序列中所有的低血压片段
        Args:
            x_future: 未来序列 [seq_len]
        Returns:
            list of tuples: 每个元组包含低血压片段的起始和结束索引
        """
        ioh_threshold = 65
        duration_threshold = 2
        segments = []
        
        # 找出所有低于阈值的点
        low_pressure_points = x_future < ioh_threshold
        
        start_idx = None
        for i in range(len(x_future)):
            if low_pressure_points[i]:
                if start_idx is None:
                    start_idx = i
                # 检查是否达到持续时间
                if i - start_idx + 1 >= duration_threshold:
                    segments.append((start_idx, i+1))
            else:
                start_idx = None
                
        return segments
    
    def sample_augment_data(self):
        """简单的数据增强方法：直接复制低血压片段来达到目标比例"""
        # 统计正负样本
        ioh_samples = []
        non_ioh_samples = []
        
        # 先统计原始样本中的正负样本比例
        for caseid, start in self.samples:
            full_data = self.case_data[caseid]['Solar8000/ART_MBP']
            end = start + self.seq_len
            label_end = end + self.pred_len
            x_future = np.array(full_data[end:label_end])
            
            is_ioh = user_definable_IOH(x_future.reshape(1, -1))[0]
            
            if is_ioh:
                ioh_samples.append((caseid, start))
            else:
                non_ioh_samples.append((caseid, start))
        
        print(f"Original IOH/non-IOH ratio: {len(ioh_samples)}/{len(non_ioh_samples)}")
        
        if len(ioh_samples) == 0:
            print("Warning: No IOH samples found!")
            return
            
        # 计算当前比例和目标比例
        current_ratio = len(ioh_samples) / max(len(non_ioh_samples), 1)
        target_ratio = self.args.augmentation_ratio
        
        if current_ratio >= target_ratio:
            return
            
        # 计算需要复制的次数
        num_augment = int(len(non_ioh_samples) * target_ratio - len(ioh_samples))
        if num_augment <= 0:
            return
            
        # 随机选择要复制的样本
        selected_samples = np.random.choice(len(ioh_samples), num_augment, replace=True)
        augmented_samples = [ioh_samples[i] for i in selected_samples]
        
        # 添加增强样本
        self.samples.extend(augmented_samples)
        final_ioh_count = len(ioh_samples) + len(augmented_samples)
        print(f"After simple augmentation - Total samples: {len(self.samples)}")
        print(f"Final IOH/non-IOH ratio: {final_ioh_count}/{len(non_ioh_samples)} = {final_ioh_count/len(non_ioh_samples):.3f}")

    def __augment_data__(self):
        """基于低血压片段的数据增强方法"""
        # 统计正负样本
        ioh_samples = []
        non_ioh_samples = []
        
        # 先统计原始样本中的正负样本比例
        for caseid, start in self.samples:
            full_data = self.case_data[caseid]['Solar8000/ART_MBP']
            end = start + self.seq_len
            label_end = end + self.pred_len
            x_future = np.array(full_data[end:label_end])
            
            is_ioh = user_definable_IOH(x_future.reshape(1, -1))[0]
            
            if is_ioh:
                ioh_samples.append((caseid, start))
            else:
                non_ioh_samples.append((caseid, start))
        
        print(f"Original IOH/non-IOH ratio: {len(ioh_samples)}/{len(non_ioh_samples)}")
        
        if len(ioh_samples) == 0:
            print("Warning: No IOH samples found!")
            return 
            
        # 计算当前比例和目标比例
        current_ratio = len(ioh_samples) / max(len(non_ioh_samples), 1)
        target_ratio = self.args.augmentation_ratio
        
        if current_ratio >= target_ratio:
            return
            
        # 对每个IOH样本进行增强
        augmented_samples = []
        for caseid, start in ioh_samples:
            full_data = self.case_data[caseid]['Solar8000/ART_MBP']
            end = start + self.seq_len
            label_end = end + self.pred_len
            x_future = np.array(full_data[end:label_end])
            
            # 找出低血压片段
            segments = self._find_ioh_segments(x_future)
            for start_idx, end_idx in segments:
                # 1. 以低血压片段开始的样本
                new_start = start + start_idx
                if new_start >= 0 and new_start + self.seq_len + self.pred_len <= len(full_data):
                    augmented_samples.append((caseid, new_start))
                
                # 2. 以低血压片段结束的样本
                new_start = start + end_idx - self.pred_len
                if new_start >= 0 and new_start + self.seq_len + self.pred_len <= len(full_data):
                    augmented_samples.append((caseid, new_start))
                
                # 3. 低血压片段在中间的样本
                mid_point = (start_idx + end_idx) // 2
                new_start = start + mid_point - self.pred_len//2
                if new_start >= 0 and new_start + self.seq_len + self.pred_len <= len(full_data):
                    augmented_samples.append((caseid, new_start))
                
                # 检查是否已达到目标比例
                if len(augmented_samples) + len(ioh_samples) >= len(non_ioh_samples) * target_ratio:
                    break
            
            if len(augmented_samples) + len(ioh_samples) >= len(non_ioh_samples) * target_ratio:
                break
        
        # 添加增强样本
        self.samples.extend(augmented_samples)
        final_ioh_count = len(ioh_samples) + len(augmented_samples)
        print(f"After complex augmentation - Total samples: {len(self.samples)}")
        print(f"Final IOH/non-IOH ratio: {final_ioh_count}/{len(non_ioh_samples)} = {final_ioh_count/len(non_ioh_samples):.3f}")

    def __getitem__(self, index):
        caseid, start = self.samples[index]
        full_data = self.case_data[caseid]
        
        # 计算实际位置
        end = start + self.seq_len
        label_end = end + self.pred_len
        
        # 获取数据
        if self.features == 'S':  # 单变量模式
            x_context = np.array(full_data['Solar8000/ART_MBP'][start:end])
            # 获取label和prediction部分
            label = np.array(full_data['Solar8000/ART_MBP'][end-self.label_len:end])
            prediction = np.array(full_data['Solar8000/ART_MBP'][end:label_end])
            # 拼接label和prediction
            x_future = np.concatenate([label, prediction])
            # 添加维度
            x_context = x_context.reshape(-1, 1)  # (seq_len, 1)
            x_future = x_future.reshape(-1, 1)    # (label+pred_len, 1)
        else:  # 多变量模式
            x_context_list = []
            for feature in self.time_series_features:  # 保持特征顺序与列表定义一致
                x_context_list.append(np.array(full_data[feature][start:end]))
            # 堆叠成多变量数据，axis=1确保特征维度按列排列
            x_context = np.stack(x_context_list, axis=1)  # (seq_len, features)
            
            # 获取label和prediction部分，同样保持特征顺序
            x_future_list = []
            for feature in self.time_series_features:  # 保持特征顺序与列表定义一致
                label = np.array(full_data[feature][end-self.label_len:end])
                prediction = np.array(full_data[feature][end:label_end])
                x_future_list.append(np.concatenate([label, prediction]))
            # 堆叠成多变量数据，axis=1确保特征维度按列排列
            x_future = np.stack(x_future_list, axis=1)  # (label+pred_len, features)
        
        # 生成时间戳特征
        seq_x_mark = self._generate_time_features(x_context)
        seq_y_mark = self._generate_time_features(x_future)

        return x_context, x_future, seq_x_mark, seq_y_mark

    def _generate_time_features(self, data):
        """生成时间戳特征"""
        time_steps = len(data)
        if self.timeenc == 0:
            time_features = np.zeros((time_steps, 4))  # month, day, weekday, hour
            for i in range(time_steps):
                time_features[i, 0] = 1 
                time_features[i, 1] = 1
                time_features[i, 2] = 1
                time_features[i, 3] = 1
        else:
            time_features = np.zeros((time_steps, 4))
            for i in range(time_steps):
                time_features[i, 0] = 1
                time_features[i, 1] = 1
                time_features[i, 2] = 1
                time_features[i, 3] = 1
        return time_features

    def __len__(self):
        return len(self.samples)

    def inverse_transform(self, data):
        """逆归一化"""
        if self.features == 'S':  # 单变量模式
            return self.scalers['Solar8000/ART_MBP'].inverse_transform(data)
        else:  # 多变量模式
            # 检查数据维度
            if data.shape[1] == 1:  # 如果数据只有1列，按单变量处理
                return self.scalers['Solar8000/ART_MBP'].inverse_transform(data)
            else:  # 多变量处理
                # 特征顺序为:
                # 0: 'Solar8000/BT' - 体温
                # 1: 'Solar8000/HR' - 心率
                # 2: 'Solar8000/ART_DBP' - 动脉舒张压
                # 3: 'Solar8000/ART_MBP' - 动脉平均血压 (目标变量)
                result = np.zeros_like(data)
                for i, feature in enumerate(self.time_series_features):  # 保持与特征列表相同的顺序
                    result[:, i] = self.scalers[feature].inverse_transform(data[:, i].reshape(-1, 1)).flatten()
                return result

    def _get_train_scaler(self):
        return self.scalers

    def get_all_caseids(self, include_sets=['test']):
        """获取所有病例ID"""
        if include_sets is None:
            include_sets = [self.flag]
        
        caseids = set()
        for caseid in self.case_data.keys():
            caseids.add(caseid)
        
        return list(caseids)

    def get_case_data_length(self, caseid):
        """获取指定病例的数据长度"""
        if caseid not in self.case_data:
            raise ValueError(f"Case ID {caseid} not found in dataset")
        return len(self.case_data[caseid]['Solar8000/ART_MBP'])

    def get_case_valid_range(self, caseid):
        """获取指定病例的有效数据范围"""
        if caseid in self.case_data:
            return (0, len(self.case_data[caseid]['Solar8000/ART_MBP']))
        return None

    def get_case_series(self, caseid, start_idx=None, end_idx=None):
        """获取指定病例的时间序列数据，返回所有特征的数据"""
        if caseid not in self.case_data:
            raise ValueError(f"Case ID {caseid} not found in dataset")
        
        if start_idx is None:
            start_idx = 0
        if end_idx is None: # 如果init的scale为True, 则后续的数据都是归一化后的数据
            end_idx = len(self.case_data[caseid]['Solar8000/ART_MBP'])
            
        # 获取所有特征的数据
        data_dict = {}
        for feature in self.time_series_features:
            if feature in self.case_data[caseid]:
                data_dict[feature] = np.array(self.case_data[caseid][feature][start_idx:end_idx])
                
        return data_dict

    def create_case_dataloader(self, caseid, batch_size, seq_len, pred_len, sample_step=1):
        """为指定的病例创建数据加载器，用于TTT，支持多变量数据"""
        range_info = self.get_case_valid_range(caseid)
        if range_info is None:
            raise ValueError(f"Case ID {caseid} not found in current {self.flag} set")
            
        start_idx, end_idx = range_info
        data_dict = self.get_case_series(caseid, start_idx, end_idx)
        
        # 数据长度验证
        required_length = seq_len + pred_len
        if len(data_dict['Solar8000/ART_MBP']) < required_length:
            return []
        
        # 生成该病例的所有样本
        batches = []
        contexts_batch = []
        futures_batch = []
        x_marks_batch = []
        y_marks_batch = []

        for i in range(0, len(data_dict['Solar8000/ART_MBP']) - required_length + 1, sample_step):
            segment_data = data_dict['Solar8000/ART_MBP'][i:i+required_length]
            
            if self.scale:
                segment_data_original = self.scalers['Solar8000/ART_MBP'].inverse_transform(
                    segment_data.reshape(-1, 1)).flatten()
                # 使用原始数据的阈值(60 mmHg)检测突变
                if (np.abs(np.diff(segment_data_original)) > 60).any():
                    continue  # abrupt change -> noise
            else:
                # 如果数据未归一化,直接使用原始阈值
                if (np.abs(np.diff(segment_data)) > 60).any():
                    continue  # abrupt change -> noise

            # 准备多变量输入数据
            # 特征顺序为:
            # 0: 'Solar8000/BT' - 体温
            # 1: 'Solar8000/HR' - 心率
            # 2: 'Solar8000/ART_DBP' - 动脉舒张压
            # 3: 'Solar8000/ART_MBP' - 动脉平均血压 (目标变量)
            x_context_list = []
            for feature in self.time_series_features:  # 严格按照特征列表定义的顺序
                if feature in data_dict:
                    x_context_list.append(torch.tensor(data_dict[feature][i:i+seq_len], dtype=torch.float32))
            x_context = torch.stack(x_context_list, dim=1)  # [seq_len, features]
            
            # 准备多变量未来数据，同样保持特征顺序一致
            x_future_list = []
            for feature in self.time_series_features:  # 严格按照特征列表定义的顺序
                if feature in data_dict:
                    x_future_list.append(torch.tensor(data_dict[feature][i+seq_len-self.label_len:i+seq_len+pred_len], dtype=torch.float32))
            x_future = torch.stack(x_future_list, dim=1)  # [label+pred_len, features]
            
            # 生成时间特征并转换为张量
            x_mark_np = self._generate_time_features(x_context.numpy())
            y_mark_np = self._generate_time_features(x_future.numpy())
            x_mark = torch.tensor(x_mark_np, dtype=torch.float32)
            y_mark = torch.tensor(y_mark_np, dtype=torch.float32)
            
            contexts_batch.append(x_context)
            futures_batch.append(x_future)
            x_marks_batch.append(x_mark)
            y_marks_batch.append(y_mark)
            
            if len(contexts_batch) == batch_size:
                batches.append((
                    torch.stack(contexts_batch),  # [batch_size, seq_len, features]
                    torch.stack(x_marks_batch),   # [batch_size, seq_len, 4]
                    torch.stack(y_marks_batch),   # [batch_size, label+pred_len, 4]
                    torch.stack(futures_batch)    # [batch_size, label+pred_len, features]
                ))
                
                # 清空列表，准备下一个批次
                contexts_batch = []
                futures_batch = []
                x_marks_batch = []
                y_marks_batch = []
        
        # 处理最后一个不完整的batch
        if contexts_batch:
            batches.append((
                torch.stack(contexts_batch),
                torch.stack(x_marks_batch),
                torch.stack(y_marks_batch),
                torch.stack(futures_batch)
            ))
            
        return batches