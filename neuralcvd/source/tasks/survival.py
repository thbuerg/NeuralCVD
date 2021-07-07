import random

import pandas as pd

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, MultiStepLR
import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data
from tqdm.auto import tqdm
from copy import deepcopy
from omegaconf.listconfig import ListConfig

from sksurv.metrics import concordance_index_ipcw
from lifelines.utils import concordance_index
from sklearn.isotonic import IsotonicRegression

from pycox.models.loss import nll_pmf_cr, rank_loss_deephit_cr
from pycox.models.utils import pad_col

from neuralcvd.source.losses.survival import *
from neuralcvd.source.evaluation.survival import get_observed_probability
from neuralcvd.source.datamodules.datasets import BatchedDS, DeepHitBatchedDS


class AbstractSurvivalTask(pl.LightningModule):
    """
    Defines a Task (in the sense of Lightning) to train a CoxPH-Model.
    """

    def __init__(self, network,
                 transforms=None,
                 batch_size=128,
                 num_workers=8,
                 lr=1e-3,
                 evaluation_time_points=[6, 11],
                 evaluation_quantile_bins=None,
                 report_train_metrics=True,
                 optimizer=torch.optim.SGD,
                 optimizer_kwargs={},
                 schedule=None,
                 schedule_kwargs={}, **kwargs):
        """
        Defines a Task (in the sense of Lightning) to train a CoxPH-Model.

        :param network: `nn.Module or pl.LightningModule`,  the network that should be used.
        :param transforms:  `nn.Module or pl.LightningModule`, optional contains the Transforms applied to input.
        :param batch_size:  `int`, batchsize
        :param num_workers: `int`, num_workers for the DataLoaders
        :param optimizer:   `torch.optim`, class, is instantiated w/ the passed optimizer args by trainer.
        :param optimizer_kwargs:    `dict`, optimizer args.
        :param schedule:    `scheudle calss` to use
        :param schedule_kwargs:  `dict`, schedule kwargs, like: {'patience':10, 'threshold':0.0001, 'min_lr':1e-6}
        """
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.net = network
        self.transforms = transforms

        self.lr = lr
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        self.schedule = schedule
        self.schedule_kwargs = schedule_kwargs

        self.evaluation_quantile_bins = eval(evaluation_quantile_bins) if not isinstance(evaluation_quantile_bins, (list, ListConfig)) else evaluation_quantile_bins
        self.evaluation_time_points = eval(evaluation_time_points) if not isinstance(evaluation_time_points, (list, ListConfig)) else evaluation_time_points

        self.params = []
        self.networks = [self.net]
        for n in self.networks:
            if n is not None:
                self.params.extend(list(n.parameters()))

        # save the params.
        self.save_hyperparameters()

    def unpack_batch(self, batch):
        data, (durations, events) = batch
        return data, durations, events

    def configure_optimizers(self):
        if isinstance(self.optimizer, str): self.optimizer = eval(self.optimizer)
        if isinstance(self.schedule, str): self.schedule = eval(self.schedule)
        self.optimizer_kwargs["lr"] = self.lr

        optimizer = self.optimizer(self.params, **self.optimizer_kwargs)
        print(f'Using Optimizer {str(optimizer)}.')
        if self.schedule is not None:
            print(f'Using Scheduler {str(self.schedule)}.')
            schedule = self.schedule(optimizer, **self.schedule_kwargs)
            if isinstance(self.schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                return {
                    'optimizer': optimizer,
                    'scheduler': schedule,
                    'monitor': 'Ctd_0.9',
                }
            else:
                return [optimizer], [schedule]
        else:
            print('No Scheduler specified.')
            return optimizer

    def ext_dataloader(self, ds, batch_size=None, shuffle=False, num_workers=None,
                       drop_last=False):  ### Already transformed datamodules? -> Or pass transformers?
        if batch_size is None:
            batch_size = self.batch_size
        if num_workers is None:
            num_workers = self.num_workers
        return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                          shuffle=shuffle, drop_last=drop_last)

    def training_step(self, batch, batch_idx):
        data, durations, events = self.unpack_batch(batch)
        if self.transforms is not None:
            data = self.transforms.apply_train_transform(data, events)
        args = self.shared_step(data, durations, events)
        loss_dict = self.loss(*args)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        data, durations, events = self.unpack_batch(batch)
        if self.transforms is not None:
            data = self.transforms.apply_valid_transform(data)
        args = self.shared_step(data, durations, events)
        loss_dict = self.loss(*args)
        loss_dict = dict([(f'val_{k}', v) for k, v in loss_dict.items()])
        for k, v in loss_dict.items():
            self.log(k, v, on_step=False, on_epoch=False, prog_bar=False, logger=False)
        return loss_dict

    # def on_train_epoch_end(self, outputs):
    #     raise NotImplementedError('train epoch end')
    # return loss_dict['loss']

    def validation_epoch_end(self, outputs):
        metrics = {}
        # aggregate the per-batch-metrics:
        for metric_name in ["val_loss"]:
            # for metric_name in [k for k in outputs[0].keys() if k.startswith("val")]:
            metrics[metric_name] = torch.stack([output[metric_name] for output in outputs]).mean()

        # calculate the survival metrics
        valid_ds = self.val_dataloader().dataset if not \
            isinstance(self.val_dataloader().dataset, (BatchedDS, DeepHitBatchedDS)) \
            else self.val_dataloader().dataset.dataset
        train_ds = self.train_dataloader().dataset if not \
            isinstance(self.train_dataloader().dataset, (BatchedDS, DeepHitBatchedDS)) \
            else self.train_dataloader().dataset.dataset
        metrics_survival = self.calculate_survival_metrics(train_ds=train_ds, valid_ds=valid_ds,
                                                           time_points=self.evaluation_time_points,
                                                           quantile_bins = self.evaluation_quantile_bins)
        # train metrics:
        if self.hparams.report_train_metrics:
            train_metrics_survival = self.calculate_survival_metrics(train_ds=train_ds, valid_ds=train_ds,
                                                                     time_points = self.evaluation_time_points,
                                                                     quantile_bins = self.evaluation_quantile_bins)
            for key, value in train_metrics_survival.items():
                metrics[f'train_{key}'] = value

        for key, value in metrics_survival.items():
            metrics[f'valid_{key}'] = value

        for key, value in metrics.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def calculate_survival_metrics(self, train_ds, valid_ds, time_points=[6, 11,], quantile_bins=None):
        """
        Calculate epoch level survival metrics.
        :param train_ds:
        :param valid_ds:
        :param time_points: times at which to evaluate.
        :param quantile_bins: ALTERNATIVELY (to time_points) -> pass quantiles of the time axis.
        :return:
        """
        metrics = {}
        ctds = []
        cs = []

        assert None in [time_points, quantile_bins], 'EITHER pass quantiles OR pass timepoints'

        try:
            surv_train = np.stack([train_ds.events.values, train_ds.durations.values], axis=0).squeeze(axis=-1)
            surv_valid = np.stack([valid_ds.events.values, valid_ds.durations.values], axis=0).squeeze(axis=-1)
        except AttributeError:
            surv_train = np.stack([train_ds.events, train_ds.durations], axis=0).squeeze(axis=-1)
            surv_valid = np.stack([valid_ds.events, valid_ds.durations], axis=0).squeeze(axis=-1)

        # move to structured arrays:
        struc_surv_train = np.array([(e, d) for e, d in zip(surv_train[0], surv_train[1])],
                                    dtype=[('event', 'bool'), ('duration', 'f8')])
        struc_surv_valid = np.array([(e, d) for e, d in zip(surv_valid[0], surv_valid[1])],
                                    dtype=[('event', 'bool'), ('duration', 'f8')])

        self.eval()
        loader = self.ext_dataloader(valid_ds, batch_size=32, num_workers=0, shuffle=False, drop_last=False)

        if time_points is None:
            assert quantile_bins is not None, 'If timepoints is None, then pass quantile bins'
            taus = [np.quantile(surv_valid[1, surv_valid[0] > 0], q) for q in quantile_bins]
            annot = quantile_bins
        else:
            taus = time_points
            annot = time_points

        for i, tau in enumerate(taus):
            risks = []
            tau_ = torch.Tensor([tau])
            with torch.no_grad():
                for batch in loader:
                    data, durations, events = self.unpack_batch(batch)
                    if self.transforms is not None:
                        data = self.transforms.apply_valid_transform(data)
                    risk = self.predict_risk(data, t=tau_)  # returns RISK (e.g. F(t))
                    risks.append(risk.detach().cpu().numpy())
            try:
                risks = np.concatenate(risks, axis=0)
            except ValueError:
                risks = np.asarray(risks)
            risks = risks.ravel()
            risks[pd.isna(risks)] = np.nanmax(risks)
            Ctd = concordance_index_ipcw(struc_surv_train, struc_surv_valid,
                                       risks,
                                       tau=tau, tied_tol=1e-8)
            C = concordance_index(event_times=surv_valid[1],
                                  predicted_scores=-risks,
                                  event_observed=surv_valid[0])
            ctds.append(Ctd[0])
            cs.append(C)

        self.train()

        for k, v in zip(annot, ctds):
            metrics[f'Ctd_{k}'] = v
        for k, v in zip(annot, cs):
            metrics[f'C_{k}'] = v

        return metrics

    def shared_step(self, data, duration, events):
        """
        shared step between training and validation. should return a tuple that fits in loss.
        :param data:
        :param durations:
        :param events:
        :return:
        """
        raise NotImplementedError("Abstract method")
        return durations, events, some_args

    def loss(self, predictions, durations, events):
        """
        Calculate Loss.
        :param predictions:
        :param durations:
        :param events:
        :return:
        """
        raise NotImplementedError("Abstract Class")
        loss1 = None
        loss2 = None
        loss = None
        return {'loss': loss,
                'loss1': loss1,
                'loss2': loss2,}

    def predict_dataset(self, ds, times):
        """
        Predict the survival function for each sample in the dataset at all durations in the dataset.

        Returns a pandas DataFrame where the rows are timepoints and the columns are the samples. Values are S(t|X)
        :param ds:
        :param times: a np.Array holding the times for which to calculate the risk.
        :return:
        """
        raise NotImplementedError("Abstract method")

    def fit_isotonic_regressor(self, ds, times, n_samples):
        if len(ds) < n_samples: n_samples = len(ds)
        sample_idx = random.sample([i for i in range(len(ds))], n_samples)
        sample_ds = torch.utils.data.Subset(ds, sample_idx)
        pred_df = self.predict_dataset(sample_ds, np.array(times)).dropna()
        for t in times:
            if hasattr(self, 'isoreg'):
                if f"0_{t}_Ft" in self.isoreg:
                    pass
                else:
                    for i, array in enumerate([pred_df[f"0_{t}_Ft"].values, pred_df["events"].values, pred_df["durations"].values]):
                        if len(list(np.argwhere(np.isnan(array))))>0:
                            print(i)
                            print(np.argwhere(np.isnan(array)))
                    F_t_obs, nan = get_observed_probability(pred_df[f"0_{t}_Ft"].values, pred_df["events"].values, pred_df["durations"].values, t)
                    self.isoreg[f"0_{t}_Ft"] = IsotonicRegression().fit(pred_df.drop(pred_df.index[nan])[f"0_{t}_Ft"].values, F_t_obs)
            else:
                F_t_obs, nan = get_observed_probability(pred_df[f"0_{t}_Ft"].values, pred_df["events"].values, pred_df["durations"].values, t)
                self.isoreg = {f"0_{t}_Ft": IsotonicRegression().fit(pred_df.drop(pred_df.index[nan])[f"0_{t}_Ft"].values, F_t_obs)}

    def predict_dataset_calibrated(self, ds, times):
        pred_df = self.predict_dataset(ds, np.array(times))
        for t in times:
            pred_df[f"0_{t}_Ft"] = self.isoreg[f"0_{t}_Ft"].predict(pred_df[f"0_{t}_Ft"])
        return pred_df

    @auto_move_data
    def forward(self, X, t=None):
        """
        Predict a sample
        :return: f_t, F_t, S_t
        """
        raise NotImplementedError("Abstract method")

    @auto_move_data
    def predict_risk(self, X, t=None):
        """
        Predict risk for X. Risk and nothing else.

        :param X:
        :param t:
        :return:
        """
        raise NotImplementedError("Abstract method")


class AbstractCompetingRisksSurvivalTask(AbstractSurvivalTask):
    """
    ABC for competing risks tranings.
    """
    def __init__(self, network,
                 transforms=None,
                 n_events=1,
                 event_names=None,
                 batch_size=128,
                 num_workers=8,
                 evaluation_time_points=[6, 11],
                 evaluation_quantile_bins=None,
                 lr=1e-3,
                 report_train_metrics=True,
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs={},
                 schedule=None,
                 schedule_kwargs={},
                 **kwargs):
        """
        Abstract class for survival prediction with competing risks.

        :param network:
        :param transforms:
        :param n_events:    `int`, number of competing events. Minimum 1.
        :param event_names: `list`,  list of str, len() = n_events -> replaces names in logs and reported metricslist of str, same length as targets (=1) -> replaces names in logs and reported metrics.
        :param batch_size:
        :param num_workers:
        :param optimizer:
        :param optimizer_kwargs:
        :param schedule:    `Schedule class`,   will be instantiated by trainer.
        :param schedule_kwargs:  `dict`, schedule kwargs, like: {'patience':10, 'threshold':0.0001, 'min_lr':1e-6}
        """
        super().__init__(
            network=network,
            transforms=transforms,
            num_workers=num_workers,
            batch_size=batch_size,
            evaluation_time_points=evaluation_time_points,
            evaluation_quantile_bins=evaluation_quantile_bins,
            lr=lr,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            schedule=schedule,
            schedule_kwargs=schedule_kwargs)

        self.n_events = n_events
        if event_names is not None:
            assert len(event_names) == self.n_events, 'Nr of event_names passed needs to match nr of competing events.'
            self.event_names = event_names
        else:
            self.event_names = [f'event_{i}' for i in range(1, self.n_events+1)]

    @auto_move_data
    def predict_risk(self, X):
        """
        Predict __RISK__ for X.

        :param X:
        :return:
        """
        raise NotImplementedError()

    def calculate_survival_metrics(self, train_ds, valid_ds, time_points=[6, 11], quantile_bins=None):
        """
        THIS IS THE COMPETING EVENTS VERSION!

        1. Calculate the Ctd on the quartiles of the valid set.
        2. Calculate the Brier scires for the same times.
        :return:
        """
        metrics = {}

        assert any([t in [time_points, quantile_bins] for t in ['None', 'none', None]]), 'EITHER pass quantiles OR pass timepoints'

        try:
            surv_train = np.stack([train_ds.events.values, train_ds.durations.values], axis=0).squeeze(axis=-1)
            surv_valid = np.stack([valid_ds.events.values, valid_ds.durations.values], axis=0).squeeze(axis=-1)
        except AttributeError:
            surv_train = np.stack([train_ds.events, train_ds.durations], axis=0).squeeze(axis=-1)
            surv_valid = np.stack([valid_ds.events, valid_ds.durations], axis=0).squeeze(axis=-1)

        self.eval()
        loader = self.ext_dataloader(valid_ds, batch_size=32, num_workers=0, shuffle=False, drop_last=False)

        if time_points is None:
            assert quantile_bins is not None, 'If timepoints is None, then pass quantile bins'
            taus = [np.quantile(surv_valid[1, surv_valid[0] > 0], q) for q in quantile_bins]
            annot = quantile_bins
        else:
            taus = time_points
            annot = time_points

        for i, tau in enumerate(taus):
            risks = []
            tau_ = torch.Tensor([tau])
            with torch.no_grad():
                for batch in loader:
                    data, d, e = self.unpack_batch(batch)

                    if self.transforms is not None:
                        data = self.transforms.apply_valid_transform(data)
                    risk = self.predict_risk(data, t=tau_)
                    del data
                    risks.append(risk.detach().cpu().numpy())

            risks = np.concatenate(risks, axis=1)

            c_per_event = []
            ctd_per_event = []
            for e in range(1, self.n_events + 1):
                e_risks = risks[e-1].ravel()
                e_risks[pd.isna(e_risks)] = np.nanmax(e_risks)
                # move to structured arrays:
                struc_surv_train = np.array([(1 if e_ == e else 0, d) for e_, d in zip(surv_train[0], surv_train[1])],
                                            dtype=[('event', 'bool'), ('duration', 'f8')])
                struc_surv_valid = np.array([(1 if e_ == e else 0, d) for e_, d in zip(surv_valid[0], surv_valid[1])],
                                            dtype=[('event', 'bool'), ('duration', 'f8')])

                Ctd = concordance_index_ipcw(struc_surv_train, struc_surv_valid,
                                           e_risks,
                                           tau=tau, tied_tol=1e-8)
                C = concordance_index(event_times=surv_valid[1],
                                      predicted_scores=-risks,
                                      event_observed=surv_valid[0])
                ctd_per_event.append(Ctd[0])
                c_per_event.append(C)
                metrics[f'{self.event_names[e-1]}_Ctd_{annot[i]}'] = Ctd[0]
                metrics[f'{self.event_names[e-1]}_C_{annot[i]}'] = C

            metrics[f'Ctd_{annot[i]}'] = np.asarray(ctd_per_event).mean()
            metrics[f'C_{annot[i]}'] = np.asarray(c_per_event).mean()

        self.train()
        return metrics

    def fit_isotonic_regressor(self, ds, times, n_samples):
        if len(ds) < n_samples: n_samples = len(ds)
        sample_idx = random.sample([i for i in range(len(ds))], n_samples)
        sample_ds = torch.utils.data.Subset(ds, sample_idx)
        pred_df = self.predict_dataset(sample_ds, np.array(times)).dropna()
        for t in tqdm(times):
            if hasattr(self, 'isoreg'):
                if f"0_{t}_Ft" in self.isoreg:
                    pass
                else:
                    for i, array in enumerate([pred_df[f"0_{t}_Ft"].values, pred_df["events"].values, pred_df["durations"].values]):
                        if len(list(np.argwhere(np.isnan(array))))>0:
                            print(i)
                            print(np.argwhere(np.isnan(array)))
                    F_t_obs, nan = get_observed_probability(pred_df[f"0_{t}_Ft"].values, pred_df["events"].values, pred_df["durations"].values, t)
                    self.isoreg[f"0_{t}_Ft"] = IsotonicRegression().fit(pred_df.drop(
                        pred_df.index[nan])[f"0_{t}_Ft"].reset_index(drop=True).values, F_t_obs.values)
            else:
                F_t_obs, nan = get_observed_probability(pred_df[f"0_{t}_Ft"].values, pred_df["events"].values, pred_df["durations"].values, t)
                self.isoreg = {f"0_{t}_Ft": IsotonicRegression().fit(
                    pred_df.drop(pred_df.index[nan])[f"0_{t}_Ft"].reset_index(drop=True).values, F_t_obs.values)}

    def predict_dataset_calibrated(self, ds, times):
        pred_df = self.predict_dataset(ds, np.array(times))
        for t in times:
            pred_df[f"0_{t}_Ft_native"] = pred_df[f"0_{t}_Ft"]
            pred_df[f"0_{t}_Ft"] = self.isoreg[f"0_{t}_Ft"].predict(pred_df[f"0_{t}_Ft_native"])
        return pred_df

    def predict_dataset(self, ds, times):
        raise NotImplementedError('Abstract.')


class DeepSurvivalMachine(AbstractCompetingRisksSurvivalTask):
    """
    pl.Lightning Module for DeepSurvivalMachines.
    """
    def __init__(self,
                 network=None,
                 transforms=None,
                 n_events=1,
                 event_names=None,
                 batch_size=1024,
                 alpha=1,
                 gamma=1e-8,
                 k_dim=8,
                 output_dim=100,
                 temperature=1000,
                 network_shape=None,
                 network_scale=None,
                 network_ks=None,
                 distribution='weibull',
                 num_workers=8,
                 lr=1e-2,
                 report_train_metrics=True,
                 evaluation_time_points=[11],
                 evaluation_quantile_bins=None,
                 optimizer=torch.optim.SGD,
                 optimizer_kwargs={},
                 schedule=None,
                 schedule_kwargs={},
                 lambda_sparse=None,
                 **kwargs
                 ):

        """
        pl.Lightning Module for DeepSurvivalMachines.
        :param network:          `nn.Module` or `pl.LightningModule`, the network
        :param transforms:       `nn.Module` holding the transformations to apply to data if necessary
        :param n_events:         `int`, number of events to consider. Minimum 1.
        :param batch_size:       `int`, batchsize
        :param alpha:            `float`, ]0, 1] weight for the loss function (1 = equal ratio, <1, upweighting of f_t(X))
        :param gamma:            `float`, factor to add to the shape and scale params to avoid edge conditions.
        :param k_dim:            `int`, number of distributions in the mixture
        :param output_dim:       `int`, outdim of `network` and in_dim to the layers generating k and `
        :param temperature:      `int`, temperature of the softmax parameter
        :param network_scale:    `nn.Module` or `pl.LightningModule`, the network to be used to compute the scale param,
                                    optional, if None, will put in linear layer.
        :param network_shape:    `nn.Module` or `pl.LightningModule`, the network to be used to compute the shape param,
                                    optional, if None, will put in linear layer.
        :param network_ks:       `nn.Module` or `pl.LightningModule`, the network to be used to compute the ks param,
                                    optional, if None, will put in linear layer.
        :param distribution:     `str` [`weibull`, `lognormal`] the base distribution
        :param num_workers:      `int` nr of workers for the dataloader
        :param optimizer:        `torch.optim` class, the optimizer to apply
        :param optimizer_kwargs: `dict` kwargs for optimizer
        :param schedule:         `LRschedule` class to use, optional
        :param schedule_kwargs:  `dict` kwargs for scheduler
        :param lambda_sparse:    `float` 1e-3 or lower; multiplier for the sparsity loss when using tabnet
        """
        if network is None:
            raise ValueError('You need to pass a network.')
        super().__init__(
            network=network,
            transforms=transforms,
            n_events=n_events,
            num_workers=num_workers,
            batch_size=batch_size,
            lr=lr,
            evaluation_time_points=evaluation_time_points,
            evaluation_quantile_bins=evaluation_quantile_bins,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            schedule=schedule,
            schedule_kwargs=schedule_kwargs)

        self.lambda_sparse = lambda_sparse
        self.temperature = temperature
        self.output_dim = output_dim
        self.k_dim = k_dim
        self.alpha = alpha
        self.gamma = gamma
        assert distribution in ['weibull', 'lognormal'], 'Currently only `lognormal` & `weibull` available.'
        self.distribution = distribution

        # build the nets:
        self.scale = network_scale.to(self.device) if network_scale is not None else \
            nn.Sequential(nn.Linear(self.output_dim, self.k_dim * self.n_events, bias=True),
                          nn.Softplus()
                          ).to(self.device)
        self.shape = network_shape.to(self.device) if network_shape is not None else \
            nn.Sequential(nn.Linear(self.output_dim, self.k_dim * self.n_events, bias=True),
                               nn.Softplus()
                               ).to(self.device)
        self.ks = network_ks.to(self.device) if network_ks is not None else\
            nn.Linear(self.output_dim, self.k_dim * self.n_events, bias=True).to(self.device)

        # set params to be optimized:
        self.params = []
        self.networks = [self.net, self.scale, self.shape, self.ks]
        for n in self.networks:
            if n is not None:
                self.params.extend(list(n.parameters()))

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        self.save_hyperparameters()

    @property
    def __name__(self):
        return 'DeepSurvivalMachine'

    @auto_move_data
    def sample_eventtime_from_mixture(self, data, nsamples=100):
        """
        Sample event time from mixture.
        :param scale:
        :param shape:
        :return:
        """

        if self.net.__class__.__name__ is 'TabNet':
            features, M_loss = self.net(data)
        else:
            features = self.net(data)

        scale = self.gamma + self.scale(features).view(features.size(0), self.k_dim, -1)
        shape = self.gamma + self.shape(features).view(features.size(0), self.k_dim, -1)
        ks = self.ks(features).view(features.size(0), self.k_dim, -1) / self.temperature
        # get dists
        if self.distribution == 'lognormal':
            distribution = LogNormal(
                loc=shape, # loc (float or Tensor) – mean of log of distribution
                scale=scale, # scale (float or Tensor) – standard deviation of log of the distribution
                validate_args=True)
        elif self.distribution == 'weibull':
            distribution = Weibull(
                scale,
                shape,
                validate_args=True
            )
        else:
            raise NotImplementedError('Currently only `lognormal` & `weibull` available.')

        samples = [distribution.sample() for _ in range(nsamples)]

        samples = torch.stack(samples, axis=0) # [samples, B, k, e]
        # weighted mean
        samples = (samples * F.softmax(ks, dim=1).repeat(nsamples, 1,1,1)).sum(axis=2) / self.k_dim
        sample_stds = samples.std(axis=0) # 1, B, k
        sample_means = samples.mean(axis=0)

        return sample_means, sample_stds

    def calculate_loglikelihood_under_mixture(self, scale, shape, durations):
        """Sample from the distribution"""
        if durations.dim() < 3:
            durations = durations.unsqueeze(-1)

        try:
            if self.distribution == 'lognormal':
                distribution = LogNormal(
                    loc=shape, # loc (float or Tensor) – mean of log of distribution
                    scale=scale, # scale (float or Tensor) – standard deviation of log of the distribution
                    validate_args=True)
                logf_t = distribution.log_prob(durations)
                logF_t = torch.log(0.5 + 0.5 * torch.erf(torch.div(torch.log(durations) - shape,
                                                                   np.sqrt(2)*scale)))
                logS_t = torch.log(0.5 - 0.5 * torch.erf(torch.div(torch.log(durations) - shape,
                                                                   np.sqrt(2)*scale)))
            elif self.distribution == 'weibull':
                distribution = Weibull(
                    scale,
                    shape,
                    validate_args=True
                )
                logf_t = distribution.log_prob(durations)
                logF_t = torch.log(1-torch.exp(-torch.pow(durations.div(scale), shape)))
                logS_t = -torch.pow(durations.div(scale), shape)
            else:
                raise NotImplementedError('Currently only `lognormal` & `weibull` available.')
        except ValueError:
            raise KeyboardInterrupt('NaNs in params, aborting training.')

        return logf_t, logF_t, logS_t

    def shared_step(self, data, durations, events):
        # TabNet returns a tuple
        if self.net.__class__.__name__ is 'TabNet':
            features, M_loss = self.net(data)
        else:
            features = self.net(data)

        scale = self.gamma + self.scale(features).view(features.size(0), self.k_dim, -1)
        shape = self.gamma + self.shape(features).view(features.size(0), self.k_dim, -1)

        ks = self.ks(features).view(features.size(0), self.k_dim, -1) / self.temperature

        logf_t, logF_t, logS_t = self.calculate_loglikelihood_under_mixture(scale, shape, durations)

        if self.net.__class__.__name__ is 'TabNet':
            return durations, events, logf_t, logF_t, logS_t, scale, shape, ks, M_loss
        else:
            return durations, events, logf_t, logF_t, logS_t, scale, shape, ks

    def loss(self, durations, events, logf_t, logF_t, logS_t, scale, shape, ks, M_loss=None):
        """ Calculate total DSM loss."""
        elbo_u = 0.
        elbo_c = 0.

        for e in range(1, self.n_events+1):
            elbo_u += DSM_uncensored_loss(logf_t[:, :, e-1], ks[:, :, e-1], events, e=e)
            elbo_c += DSM_censored_loss(logS_t[:, :, e-1], ks[:, :, e-1], events, e=e)

        loss_val = elbo_u + self.alpha * elbo_c

        # Sparsity loss multiplier for TabNet
        if self.net.__class__.__name__ is 'TabNet' and self.lambda_sparse is not None:
            loss_val = loss_val - (self.lambda_sparse * M_loss)

        return {'loss': loss_val,
                'uc_loss': elbo_u,
                'c_loss': elbo_c,
                }

    def predict_dataset(self, dataset: object, times=[11]):
        """
        Predict dataset at times t
        :param ds:
        :param times:
        :return:
        """
        events, durations = [], []
        loader = self.ext_dataloader(dataset, batch_size=10000, num_workers=8, shuffle=False, drop_last=False)

        self.eval()
        # predict each sample at each duration:
        f_tX, F_tX, S_tX, t_X, t_X_std = [], [], [], [], []

        with torch.no_grad():
            for batch in (loader):
                data, d, e = self.unpack_batch(batch)
                if self.transforms is not None:
                    data = self.transforms.apply_valid_transform(data)
                durations.append(d)
                events.append(e)
                f_sample, F_sample, S_sample = [], [], []
                for t in times:
                    f_preds, F_preds, S_preds = self.forward(data,  t=torch.Tensor([t]))
                    f_sample.append(f_preds.detach().cpu())
                    F_sample.append(F_preds.detach().cpu())
                    S_sample.append(S_preds.detach().cpu())

                # sample the event time (argmax f(t))
                t_sample, t_sample_std = self.sample_eventtime_from_mixture(data) # [B, e]
                del data
                t_X.append(t_sample.permute(1, 0).detach().cpu())
                t_X_std.append(t_sample_std.permute(1, 0).detach().cpu())
                S_tX.append(torch.stack(S_sample, dim=1))
                F_tX.append(torch.stack(F_sample, dim=1))
                f_tX.append(torch.stack(f_sample, dim=1))
        t_X = torch.cat(t_X, dim=-1).numpy()  # -> [e, t, n_samples]
        t_X_std = torch.cat(t_X_std, dim=-1).numpy()  # -> [e, t, n_samples]
        S_tX = torch.cat(S_tX, dim=-1).numpy()  # -> [e, t, n_samples]
        F_tX = torch.cat(F_tX, dim=-1).numpy()
        f_tX = torch.cat(f_tX, dim=-1).numpy()

        self.train()
        pred_df = []
        for e in range(self.n_events):
            t_df = []
            for t_i, t in enumerate(times):
                df = pd.DataFrame.from_dict({
                    f"{e}_{t}_ft": f_tX[e, t_i, :].ravel(),
                    f"{e}_{t}_Ft": F_tX[e, t_i, :].ravel(),
                    f"{e}_{t}_St": S_tX[e, t_i, :].ravel()})
                t_df.append(df)
            df = pd.concat(t_df, axis=1)
            df[f'{e}_time'] = t_X[e, :].ravel()
            df[f'{e}_time_std'] = t_X_std[e, :].ravel()
            pred_df.append(df)

        pred_df = pd.concat(pred_df, axis=1)
        pred_df['durations'] = torch.cat(durations, dim=0).cpu().numpy().ravel()
        pred_df['events'] = torch.cat(events, dim=0).cpu().numpy().ravel()
        return pred_df

    @auto_move_data
    def forward(self, data, t=None):
        # TabNet returns the output and
        if self.net.__class__.__name__ is 'TabNet':
            features, M_loss = self.net(data)
        else:
            features = self.net(data)

        batchsize = features.size(0)
        if batchsize != t.size(0):
            t = t.repeat(batchsize, 1)

        scale = self.gamma + self.scale(features).view(batchsize, self.k_dim, -1)
        shape = self.gamma + self.shape(features).view(batchsize, self.k_dim, -1)

        ks = self.ks(features).view(batchsize, self.k_dim, -1) / self.temperature

        logf_t, logF_t, logS_t = self.calculate_loglikelihood_under_mixture(scale, shape, t)

        S_t = torch.exp(torch.logsumexp(logS_t.add(F.log_softmax(ks, dim=1)), 1, keepdim=True)).permute(2, 1, 0).squeeze(1)
        F_t = torch.exp(torch.logsumexp(logF_t.add(F.log_softmax(ks, dim=1)), 1, keepdim=True)).permute(2, 1, 0).squeeze(1)
        f_t = torch.exp(torch.logsumexp(logf_t.add(F.log_softmax(ks, dim=1)), 1, keepdim=True)).permute(2, 1, 0).squeeze(1)

        return f_t, F_t, S_t

    @auto_move_data
    def predict_risk(self, X, t=None):
        """
        Predict Risk (= F(t)) nothing else.
        :param X:
        :param t:
        :return:
        """
        f_t, F_t, S_t = self.forward(X, t=t)
        return F_t

