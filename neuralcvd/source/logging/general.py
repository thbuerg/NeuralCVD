from pytorch_lightning.loggers import NeptuneLogger
from typing import Any, Dict, Iterable, Optional, Union
from argparse import Namespace


class FoolProofNeptuneLogger(NeptuneLogger):
    """
    Logger that does only log params if they do not exceed the str len limit.
    """
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        for key, val in params.items():
            if len(str(val)) < 16384:
                self.experiment.set_property(f'param__{key}', val)