#!/usr/bin/env python3

import warnings

from stepin.ml.cfg_model import ConfiguredModel
from stepin.pctr.data_loader import DataLoader
from stepin.script import safe_main


def _main(config):
    warnings.filterwarnings("ignore")
    mparams = config['model_params']
    loader = DataLoader(config)
    loader.load()
    cmodel = ConfiguredModel(mparams, loader)
    cmodel.grid_search_train(mparams['grid'])


safe_main('Build column storage', _main)
