"""
Revised from https://github.com/rixwew/pytorch-fm/blob/master/torchfm/dataset/avazu.py
"""

import shutil
import struct
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

import lmdb
import numpy as np
import torch.utils.data
import os.path
from tqdm import tqdm
import pandas as pd


class AdultIncome(torch.utils.data.Dataset):
    """
    Avazu Click-Through Rate Prediction Dataset

    Dataset preparation
        Remove the infrequent features (appearing in less than threshold instances) and treat them as a single feature

    :param dataset_path: avazu train path
    :param cache_path: lmdb cache path
    :param rebuild_cache: If True, lmdb cache is refreshed
    :param min_threshold: infrequent feature threshold

    Reference
        https://www.kaggle.com/c/avazu-ctr-prediction
    """

    def __init__(self, dataset_path=None, cache_path='.adultincome', rebuild_cache=False, min_threshold=4):
        self.NUM_FEATS = 14
        self.min_threshold = min_threshold
        cache_path = os.path.dirname(dataset_path)+ "/" + cache_path
        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('create cache: failed: dataset_path is None')
            self.__build_cache(dataset_path, cache_path)
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] - 1
            self.field_dims = np.frombuffer(txn.get(b'field_dims'), dtype=np.uint32)

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            np_array = np.frombuffer(
                txn.get(struct.pack('>I', index)), dtype=np.uint32).astype(dtype=np.long)
        return np_array[1:], np_array[0]

    def __len__(self):
        return self.length

    def __build_cache(self, path, cache_path):
        feat_mapper, defaults = self.__get_feat_mapper(path)
        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            field_dims = np.zeros(len(feat_mapper), dtype=np.uint32)
            for i, fm in enumerate(feat_mapper.values()):
                field_dims[i] = len(fm) + 1
            with env.begin(write=True) as txn:
                txn.put(b'field_dims', field_dims.tobytes())
            for buffer in self.__yield_buffer(path, feat_mapper, defaults):
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        txn.put(key, value)


    def __get_feat_mapper(self, path):
        feat_cnts = defaultdict(lambda: defaultdict(int))
        data = pd.read_csv(path, skipinitialspace=True)
        # Convert income levels to integers
        income_map = {'<=50K': 0, '>50K': 1}
        data['income'] = data['income'].map(income_map)
        for col in data.columns[:-1]:  # Exclude the target column
            if data[col].dtype == 'object':  # If the feature is categorical
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
            feat_cnts[col] = defaultdict(int, data[col].value_counts().to_dict())
        feat_mapper = {col: {feat for feat, c in cnt.items() if c >= self.min_threshold} for col, cnt in feat_cnts.items()}
        feat_mapper = {col: {feat: idx for idx, feat in enumerate(cnt)} for col, cnt in feat_mapper.items()}
        defaults = {col: len(cnt) for col, cnt in feat_mapper.items()}
        return feat_mapper, defaults

    def __yield_buffer(self, path, feat_mapper, defaults, buffer_size=int(1e5)):
        item_idx = 0
        buffer = list()
        data = pd.read_csv(path, skipinitialspace=True)
        income_map = {'<=50K': 0, '>50K': 1}
        data['income'] = data['income'].map(income_map)
        for _, row in data.iterrows():
            values = row.tolist()
            np_array = np.zeros(len(data.columns), dtype=np.uint32)  # Number of columns (features + target)
            np_array[0] = int(values[-1])  # Assuming the target is in the last column
            for i in range(1, len(data.columns)):  # Number of features
                np_array[i] = feat_mapper[data.columns[i-1]].get(values[i-1], defaults[data.columns[i-1]])
            buffer.append((struct.pack('>I', item_idx), np_array.tobytes()))
            item_idx += 1
            if item_idx % buffer_size == 0:
                yield buffer
                buffer.clear()
        yield buffer
