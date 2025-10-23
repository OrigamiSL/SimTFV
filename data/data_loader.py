import os
import warnings
import copy
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, data_path='ETTh1.csv', Batch_size=16, flag='train', size=None):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.input_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.input_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.root_path = root_path
        self.data_path = data_path
        self.batch_size = Batch_size
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.input_len, 12 * 30 * 24 + 4 * 30 * 24 - self.input_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        df_value = df_data.values

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.data = self.scaler.transform(df_value)
        self.data_x = self.data[border1:border2]
        self.window_num = self.data_x.shape[0] - self.input_len - self.pred_len + 1
        if self.set_type == 0:
            self.index_list = np.arange(self.window_num)
            self.index_list = np.random.choice(self.index_list, self.window_num, replace=False)

    def __getitem__(self, index):
        if self.set_type == 0:
            seq_x = []
            num_list = np.arange(self.data_x.shape[1])
            if self.data_x.shape[1] > 100:
                num = np.random.randint(low=5 * int(np.log(self.data_x.shape[1])),
                                        high=min(
                                            max(10 * int(np.log(self.data_x.shape[1])), self.data_x.shape[1] // 4),
                                            self.data_x.shape[1]),
                                        size=1)
            else:
                num = self.data_x.shape[1]
            for idx in self.index_list[index * self.batch_size: (index + 1) * self.batch_size]:
                current_index = np.random.choice(num_list, num, replace=False)
                r_begin = idx
                r_end = r_begin + self.input_len + self.pred_len
                seq_x_temp = copy.deepcopy(self.data_x[r_begin:r_end, current_index])
                seq_x.append(seq_x_temp)
            seq_x = np.stack(seq_x, axis=0)
        else:
            r_begin = index
            r_end = r_begin + self.input_len + self.pred_len
            seq_x = self.data_x[r_begin:r_end]
        return seq_x

    def __len__(self):
        if self.set_type == 0:
            return (len(self.data_x) - self.input_len - self.pred_len + 1) // self.batch_size
        else:
            return len(self.data_x) - self.input_len - self.pred_len + 1

    def train_shuffle(self):
        if self.set_type == 0:
            self.index_list = np.random.choice(self.index_list, self.window_num, replace=False)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_min(Dataset):
    def __init__(self, root_path, data_path='ETTm1.csv', Batch_size=16, flag='train', size=None):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.input_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.input_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.root_path = root_path
        self.data_path = data_path
        self.batch_size = Batch_size
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.input_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.input_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        df_value = df_data.values

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.data = self.scaler.transform(df_value)
        self.data_x = self.data[border1:border2]
        self.window_num = self.data_x.shape[0] - self.input_len - self.pred_len + 1
        if self.set_type == 0:
            self.index_list = np.arange(self.window_num)
            self.index_list = np.random.choice(self.index_list, self.window_num, replace=False)

    def __getitem__(self, index):
        if self.set_type == 0:
            seq_x = []
            num_list = np.arange(self.data_x.shape[1])
            if self.data_x.shape[1] > 100:
                num = np.random.randint(low=5 * int(np.log(self.data_x.shape[1])),
                                        high=min(
                                            max(10 * int(np.log(self.data_x.shape[1])), self.data_x.shape[1] // 4),
                                            self.data_x.shape[1]),
                                        size=1)
            else:
                num = self.data_x.shape[1]
            for idx in self.index_list[index * self.batch_size: (index + 1) * self.batch_size]:
                current_index = np.random.choice(num_list, num, replace=False)
                r_begin = idx
                r_end = r_begin + self.input_len + self.pred_len
                seq_x_temp = copy.deepcopy(self.data_x[r_begin:r_end, current_index])
                seq_x.append(seq_x_temp)
            seq_x = np.stack(seq_x, axis=0)
        else:
            r_begin = index
            r_end = r_begin + self.input_len + self.pred_len
            seq_x = self.data_x[r_begin:r_end]
        return seq_x

    def __len__(self):
        if self.set_type == 0:
            return (len(self.data_x) - self.input_len - self.pred_len + 1) // self.batch_size
        else:
            return len(self.data_x) - self.input_len - self.pred_len + 1

    def train_shuffle(self):
        if self.set_type == 0:
            self.index_list = np.random.choice(self.index_list, self.window_num, replace=False)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, data_path='ECL.csv', Batch_size=16, flag='train', size=None):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.input_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.input_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.root_path = root_path
        self.data_path = data_path
        self.batch_size = Batch_size
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.input_len, len(df_raw) - num_test - self.input_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        df_value = df_data.values

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.data = self.scaler.transform(df_value)
        self.data_x = self.data[border1:border2]
        self.window_num = self.data_x.shape[0] - self.input_len - self.pred_len + 1
        if self.set_type == 0:
            self.index_list = np.arange(self.window_num)
            self.index_list = np.random.choice(self.index_list, self.window_num, replace=False)

    def __getitem__(self, index):
        if self.set_type == 0:
            seq_x = []
            num_list = np.arange(self.data_x.shape[1])
            if self.data_x.shape[1] > 100:
                num = np.random.randint(low=5 * int(np.log(self.data_x.shape[1])),
                                        high=min(
                                            max(10 * int(np.log(self.data_x.shape[1])), self.data_x.shape[1] // 4),
                                            self.data_x.shape[1]),
                                        size=1)
            else:
                num = self.data_x.shape[1]
            for idx in self.index_list[index * self.batch_size: (index + 1) * self.batch_size]:
                current_index = np.random.choice(num_list, num, replace=False)
                r_begin = idx
                r_end = r_begin + self.input_len + self.pred_len
                seq_x_temp = copy.deepcopy(self.data_x[r_begin:r_end, current_index])
                seq_x.append(seq_x_temp)
            seq_x = np.stack(seq_x, axis=0)
        else:
            r_begin = index
            r_end = r_begin + self.input_len + self.pred_len
            seq_x = self.data_x[r_begin:r_end]
        return seq_x

    def __len__(self):
        if self.set_type == 0:
            return (len(self.data_x) - self.input_len - self.pred_len + 1) // self.batch_size
        else:
            return len(self.data_x) - self.input_len - self.pred_len + 1

    def train_shuffle(self):
        if self.set_type == 0:
            self.index_list = np.random.choice(self.index_list, self.window_num, replace=False)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Wind(Dataset):
    def __init__(self, root_path, data_path='Wind.csv', Batch_size=16, flag='train', size=None):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.input_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.input_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.root_path = root_path
        self.data_path = data_path
        self.batch_size = Batch_size
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.input_len, len(df_raw) - num_test - self.input_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        df_raw = df_raw[cols]

        cols_data = df_raw.columns[:]
        df_data = df_raw[cols_data]
        df_value = df_data.values

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.data = self.scaler.transform(df_value)
        self.data_x = self.data[border1:border2]
        self.window_num = self.data_x.shape[0] - self.input_len - self.pred_len + 1
        if self.set_type == 0:
            self.index_list = np.arange(self.window_num)
            self.index_list = np.random.choice(self.index_list, self.window_num, replace=False)

    def __getitem__(self, index):
        if self.set_type == 0:
            seq_x = []
            num_list = np.arange(self.data_x.shape[1])
            if self.data_x.shape[1] > 100:
                num = np.random.randint(low=5 * int(np.log(self.data_x.shape[1])),
                                        high=min(
                                            max(10 * int(np.log(self.data_x.shape[1])), self.data_x.shape[1] // 4),
                                            self.data_x.shape[1]),
                                        size=1)
            else:
                num = self.data_x.shape[1]
            for idx in self.index_list[index * self.batch_size: (index + 1) * self.batch_size]:
                current_index = np.random.choice(num_list, num, replace=False)
                r_begin = idx
                r_end = r_begin + self.input_len + self.pred_len
                seq_x_temp = copy.deepcopy(self.data_x[r_begin:r_end, current_index])
                seq_x.append(seq_x_temp)
            seq_x = np.stack(seq_x, axis=0)
        else:
            r_begin = index
            r_end = r_begin + self.input_len + self.pred_len
            seq_x = self.data_x[r_begin:r_end]
        return seq_x

    def __len__(self):
        if self.set_type == 0:
            return (len(self.data_x) - self.input_len - self.pred_len + 1) // self.batch_size
        else:
            return len(self.data_x) - self.input_len - self.pred_len + 1

    def train_shuffle(self):
        if self.set_type == 0:
            self.index_list = np.random.choice(self.index_list, self.window_num, replace=False)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, data_path='PEMS08.npz', Batch_size=16, flag='train', size=None):
        if size is None:
            self.input_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.input_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.root_path = root_path
        self.data_path = data_path
        self.batch_size = Batch_size
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = np.load(os.path.join(self.root_path, self.data_path), allow_pickle=True)
        df_value = df_raw['data'][:, :, 0]
        num_train = int(len(df_value) * 0.7)
        num_test = int(len(df_value) * 0.2)
        num_vali = len(df_value) - num_train - num_test
        border1s = [0, num_train - self.input_len, len(df_value) - num_test - self.input_len]
        border2s = [num_train, num_train + num_vali, len(df_value)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.data = self.scaler.transform(df_value)
        self.data_x = self.data[border1:border2]
        self.window_num = self.data_x.shape[0] - self.input_len - self.pred_len + 1
        if self.set_type == 0:
            self.index_list = np.arange(self.window_num)
            self.index_list = np.random.choice(self.index_list, self.window_num, replace=False)

    def __getitem__(self, index):
        if self.set_type == 0:
            seq_x = []
            num_list = np.arange(self.data_x.shape[1])
            if self.data_x.shape[1] > 100:
                num = np.random.randint(low=5 * int(np.log(self.data_x.shape[1])),
                                        high=min(
                                            max(10 * int(np.log(self.data_x.shape[1])), self.data_x.shape[1] // 4),
                                            self.data_x.shape[1]),
                                        size=1)
            else:
                num = self.data_x.shape[1]
            for idx in self.index_list[index * self.batch_size: (index + 1) * self.batch_size]:
                current_index = np.random.choice(num_list, num, replace=False)
                r_begin = idx
                r_end = r_begin + self.input_len + self.pred_len
                seq_x_temp = copy.deepcopy(self.data_x[r_begin:r_end, current_index])
                seq_x.append(seq_x_temp)
            seq_x = np.stack(seq_x, axis=0)
        else:
            r_begin = index
            r_end = r_begin + self.input_len + self.pred_len
            seq_x = self.data_x[r_begin:r_end]
        return seq_x

    def __len__(self):
        if self.set_type == 0:
            return (len(self.data_x) - self.input_len - self.pred_len + 1) // self.batch_size
        else:
            return len(self.data_x) - self.input_len - self.pred_len + 1

    def train_shuffle(self):
        if self.set_type == 0:
            self.index_list = np.random.choice(self.index_list, self.window_num, replace=False)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
