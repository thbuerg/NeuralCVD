import os
import glob
import collections
from _collections import OrderedDict
import numbers, random
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

import pathlib
import requests
import h5py
import anndata as ad

import torch

from torch._six import container_abcs, string_classes, int_classes
from torch.utils.data import Dataset, DataLoader

from pycox.models.data import pair_rank_mat


class TabularDataset(Dataset):
    """
    Dataset wrapper to sit ontop of a feather file, and read specific columns
    """

    def __init__(self, data_fp, features, normalization_dict=None, eid_selection_mask=None, oversampling=None):
        super().__init__()
        """
        Create a dataset to read h5ad files.
        Currently a bit ugly as we create a pd.DataFrame holding the entire dataset. We need this to perform efficient eid selection using df.loc.
        df.loc is the perfect method to do that since it sorts the datamodules to the passed argument as well. We can thus make sure that multiple h5adDatasets are in the same order.
        :param h5ad_fp: `str`, the filepath to the h5ad that should be read.
        :param features: `list` or list-like, contains the strings to select the features to be returned from the h5ad.
        :param eid_selection_mask: `list` or list-like, optional (default `None`), contains the eids to select.
        """
        # determine wheter file to read is .csv or .feather:
        ext = os.path.splitext(data_fp)[1]
        assert ext in ['.csv', '.feather'], 'TabularDataset only supports .csv and .feather files'
        print(data_fp)
        base = pathlib.Path(data_fp).parents[2]
        description_fp = os.path.join(base, f'description{ext}')
        assert os.path.exists(description_fp), f'Description file not found in {description_fp}'

        # read datamodules:
        read_method = pd.read_feather if ext == '.feather' else pd.read_csv
        data = read_method(data_fp)
        description = read_method(description_fp)

        for f in features:
            if f not in data.columns.values:
                print(f)
        assert all([c in data.columns.values for c in features]), \
            'Not all passed features were found in datamodules file columns'

        self.features = features
        description = description.query("covariate==@self.features")
        self.eid_map = data[["eid"]+self.features].copy().astype({'eid': 'int32'}).set_index('eid')

        if eid_selection_mask is not None: #self.eid_map = self.eid_map.reset_index().query("eid == @eid_selection_map").set_index("eid")
            ## find intersection of mask and eids:
            eid_selection_mask = [int(i) for i in eid_selection_mask]  # make sure its int!
            #faulty_ids = [i for i in eid_selection_mask if i not in self.eid_map.index.values]
            eids_intersection = self.eid_map.index.intersection(eid_selection_mask)
            print(f"{len(self.eid_map)-len(eids_intersection)} eids excluded")
            self.eid_map = self.eid_map.loc[eids_intersection,:]  # make sure this is sorted.
            print(len(self.eid_map))
        # normalize values
        if normalization_dict is not None:
            self.eid_map = self.normalize_df_fixed_params(self.eid_map, normalization_dict)

        # get the idxs of categorical vars:
        self.categoricals = description.query("covariate ==@self.features").\
            query("dtype in ['category', 'bool']").covariate.values
        self.continuous = description.query("covariate ==@self.features").\
            query("dtype in ['int', 'float']").covariate.values
        self.categorical_idxs = [self.eid_map.columns.tolist().index(v) for v in self.categoricals]
        self.continuous_idxs = [self.eid_map.columns.tolist().index(v)for v in self.continuous]

        for f in self.features:
            self.eid_map[f] = self.eid_map[f].astype(float)
        del data

    def normalize_df_fixed_params(self, df, param_dict):
        """
        Normalize pd.DF column-wise.
        :param df:
        :param param_dict: 'dictionary', contains columns of the df as key and tuple (min, max) as scaling factors for spec column
        :return:
        """
        print('normalizing datamodules...')
        for key in param_dict.keys():
            assert key in df.columns
        for col in df.columns:
            if col in list(param_dict.keys()):
                min = param_dict[col][0]
                max = param_dict[col][1]
                df[col] = (df[col] - min) / (max - min + 0.00001)
        return df

    def __getitem__(self, idx):
        fts = self.eid_map.values[idx, :]
        return torch.Tensor(fts)

    def __len__(self):
        return self.eid_map.shape[0]


class BatchedDS(Dataset):
    def __init__(self, dataset, batch_size, attrs=None):
        attrs = ['durations', 'events', ] if attrs is None else attrs
        for attr in attrs:
            try:
                setattr(self, attr, getattr(dataset, attr))
            except:
                print('Dataset has not attribute %s' % attr)

        self.len = len(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return self.len // self.batch_size

    def __getitem__(self, idx):
        return self.dataset[idx*self.batch_size:idx*self.batch_size+self.batch_size]

    @staticmethod
    def default_collate(batch):
        r"""Puts each datamodules field into a tensor with outer dimension batch size"""
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.cat(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':

                return BatchedDS.default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: BatchedDS.default_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(BatchedDS.default_collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError('each element in list of batch should be of equal size')
            transposed = zip(*batch)
            return [BatchedDS.default_collate(samples) for samples in transposed]


class DatasetWrapper(Dataset):
    """
    Wrap multiple datasets (datamodules) with labels (labels).
    Assumes all passed datasets have the same order.
    """
    def __init__(self,
                 covariate_datasets,
                 label_datasets):
        """
        Wrap multiple datasets (datamodules) with labels (labels).
        Assumes all passed datasets have the same order (eid-wise).

        :param covariate_datasets:  `list-like`, should contain datasets, samples all in the same order
        :param label_dataset:  `list-like`, shoudl contain datsets
        """
        assert all(len(ds) == len(label_datasets[0])
                   for ds in covariate_datasets + label_datasets), 'datasets need to be same length'
        self.datasets = covariate_datasets
        self.label_datasets = label_datasets

    @property
    def durations(self):
        return self.label_datasets[0].eid_map.values

    @property
    def events(self):
        return self.label_datasets[1].eid_map.values

    def __len__(self):
        return len(self.label_datasets[0])

    def __getitem__(self, idx):
        # return a tuple for datasets and a tuple for whatever is in labels
        # ((dataset1, dataset2, dataset3, ..)(duration, labels))
        covariates = tuple([ds[idx] for ds in self.datasets]) if len(self.datasets) > 1 else self.datasets[0][idx]
        labels = tuple([ds[idx] for ds in self.label_datasets]) if len(self.label_datasets) > 1 else self.label_datasets[0][idx]

        return covariates, labels

