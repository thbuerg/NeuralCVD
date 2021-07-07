import os
import hydra
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, MultiStepLR
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from omegaconf import DictConfig, OmegaConf

from neuralcvd.source.tasks.survival import *
from neuralcvd.source.callbacks.survival import WriteCheckpointLogs, WritePredictionsDataFrame
from neuralcvd.source.modules.general import *
from neuralcvd.source.datamodules.survival import *
from neuralcvd.source.utils.general import get_neptune_params, set_up_neptune, get_default_callbacks

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
pd.options.mode.use_inf_as_na = True

config_path = os.path.abspath("../../source/config")


def train(FLAGS):
    if FLAGS.experiment.seed is not None: pl.seed_everything(FLAGS.experiment.seed)
    else: pl.seed_everything(23)

    Module = eval(FLAGS.experiment.module)
    Net = eval(FLAGS.experiment.net)
    DataModule = eval(FLAGS.experiment.datamodule)

    ### print configuration
    print(Module.__name__); print(Net.__name__); print(DataModule.__name__)

    # load features.yaml if necessary:
    print(FLAGS.setup.config_path)
    print(f"{FLAGS.setup.config_path}/{FLAGS.experiment.features_yaml}")
    if FLAGS.experiment.feature_set is not None:
        FLAGS.experiment.features = OmegaConf.load(os.path.join(FLAGS.setup.config_path, FLAGS.experiment.features_yaml))
    print(FLAGS.experiment.features)
    datamodule = DataModule(**FLAGS.experiment)
    datamodule.prepare_data()
    datamodule.setup("fit")

    # log features
    FLAGS.experiment.update({"train_features": datamodule.features,
                             "train_targets": datamodule.event + datamodule.duration})

    ### Select Net as configured above => works but pretty dirty right now
    net = Net(input_dim=len(datamodule.features),
              **FLAGS.experiment)

    # initialize module
    module = Module(network=net,
                    **FLAGS.experiment)

    # get logger:
    callbacks = get_default_callbacks(monitor=FLAGS.experiment.monitor) + [WriteCheckpointLogs(), WritePredictionsDataFrame()]
    print(callbacks)
    neptune_logger = set_up_neptune(**get_neptune_params(FLAGS, callbacks))

    # initialize trainer
    trainer = pl.Trainer(**FLAGS.trainer, callbacks=callbacks, logger=neptune_logger)

    if FLAGS.trainer.auto_lr_find:
        trainer.tune(model=module, datamodule=datamodule, lr_find_kwargs={"min_lr": 0.001, "num_training": 500})

    trainer.fit(module, datamodule)
    trainer.logger.experiment.stop()

    print("DONE.")


@hydra.main(config_path=config_path, config_name="survival")
def main(FLAGS: DictConfig):
    OmegaConf.set_struct(FLAGS, False)
    FLAGS.setup.config_path = config_path
    print(FLAGS.pretty())
    return train(FLAGS)


if __name__ == '__main__':
    main()



