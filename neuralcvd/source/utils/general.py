import os
import json
import torch
import torch.nn as nn
import argparse
import pytorch_lightning as pl
import numpy as np
import pandas as pd

from neuralcvd.source.logging.general import FoolProofNeptuneLogger

from blitz.utils import variational_estimator


####################################################################################################
#                        neptune                                                                   #
####################################################################################################

def set_up_neptune(project_name='debug', experiment_name='debug', params={}, tags=[], close_after_fit=False, **kwargs):
    """
    Set up a neptune logger from file.
    :param keyfile:
    :param project_name:
    :param experiment_name:
    :param params:
    :param tags:
    :param close_after_fit:
    :param kwargs:
    :return:
    """
    if not "NEPTUNE_API_TOKEN" in os.environ:
        raise EnvironmentError('Please set environment variable `NEPTUNE_API_TOKEN`.')

    neptune_logger = FoolProofNeptuneLogger(api_key=os.environ["NEPTUNE_API_TOKEN"],
                                               project_name=project_name,
                                               experiment_name=experiment_name,
                                               params=params,
                                               tags=tags,
                                               close_after_fit=close_after_fit)

    return neptune_logger


def get_neptune_params(FLAGS, callbacks=[], **kwargs):
    """
    :param FLAGS:
    :param callbacks:
    :return:
    """
    neptune_params = {
        "project_name": FLAGS.setup.project_name,
        "experiment_name": FLAGS.setup.experiment_name,
        "tags": FLAGS.setup.tags.rstrip(',').split(','),
        "params": {**FLAGS.experiment, **FLAGS.trainer},
        "callbacks": [type(cb).__name__ for cb in callbacks],
        "close_after_fit": False
    }
    return neptune_params


def get_default_callbacks(monitor='Ctd_0.9', mode='max', early_stop=True):
    """
    Instantate the default callbacks: EarlyStopping and Checkpointing.

    :param monitor:
    :param mode:
    :return:
    """
    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor=monitor, verbose=True,
                                                                        save_last=True, save_top_k=3,
                                                                        save_weights_only=False, mode=mode,
                                                                        period=1)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step', log_momentum=False)
    if early_stop:
        early_stop = pl.callbacks.early_stopping.EarlyStopping(monitor=monitor, min_delta=1e-5, patience=25,
                                                               verbose=True, mode=mode, strict=False)
        return [checkpoint_callback, early_stop, lr_monitor]
    else:
        return [checkpoint_callback, lr_monitor]

