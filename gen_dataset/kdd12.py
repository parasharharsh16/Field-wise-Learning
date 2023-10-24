import shutil
import struct
from collections import defaultdict
from pathlib import Path
import lmdb
import numpy as np
import torch.utils.data
import os.path
from tqdm import tqdm
import pandas as pd

class MovieLens1MDataset(torch.utils.data.Dataset):

    def __init__(self, ratings_path, movies_path, cache_path='.movielens', rebuild_cache=False, min_threshold=4):
        self.min_threshold = min_threshold
        cache_path = os.path.dirname(ratings_path) + "/" + cache_path
        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if ratings_path is None or movies_path is None:
                raise ValueError('create cache: failed: ratings_path or movies_path is None')
            self.__build_cache(ratings_path, movies_path, cache_path)
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
    
    def __get_feat_mapper(self, path):
        feat_cnts = defaultdict(lambda: defaultdict(int))
        with open(path) as f:
            f.readline()
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create 1m dataset cache: counting features')
            for line in pbar:
                values = line.rstrip('\n').split(',')
                if len(values) != self.NUM_FEATS + 2:
                    continue
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i + 1]] += 1
        feat_mapper = {i: {feat for feat, c in cnt.items() if c >= self.min_threshold} for i, cnt in feat_cnts.items()}
        feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)} for i, cnt in feat_mapper.items()}
        defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}
        return feat_mapper, defaults


    def __build_cache(self, ratings_path, movies_path, cache_path):
        ratings_df = pd.read_csv(ratings_path, sep='::', engine='python', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
        movies_df = pd.read_csv(movies_path, sep='::', engine='python', header=None, names=['movie_id', 'title', 'genres'])
        __get_feat_mapper

        # Define your feature processing logic here
        # ...

        # Example: feat_mapper, defaults = self.__get_feat_mapper(ratings_df)

        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            field_dims = np.zeros(22, dtype=np.uint32)  # Adjust based on your dataset
            for i, fm in feat_mapper.items():
                field_dims[i - 1] = len(fm) + 1
            with env.begin(write=True) as txn:
                txn.put(b'field_dims', field_dims.tobytes())

            for buffer in self.__yield_buffer(ratings_df, feat_mapper, defaults):
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        txn.put(key, value)
    
    

    # Define your feature processing logic specific to MovieLens dataset
    # For example, __get_feat_mapper and __yield_buffer functions

    # def __get_feat_mapper(self, df):
    #     ...

    # def __yield_buffer(self, df, feat_mapper, defaults, buffer_size=int(1e5)):
    #     ...

    # Define any other dataset-specific functions or preprocessing steps

# Example usage:

ratings_file = 'path_to_your_ratings_file/ratings.dat'
movies_file = 'path_to_your_movies_file/movies.dat'

dataset = MovieLens1MDataset(ratings_file, movies_file)
