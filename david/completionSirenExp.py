#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the EMICS package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Completion Experiment
====================
"""

from __future__ import print_function
from pathlib import Path

import numpy as np
import sys

import emics.util as emu
import emics.experiment as ep
# from tensor import lrd
# from tensor import cbpdn
# import tensor.tensorSetup as ts
import sirenSetup as ss

# from sporco.util import u

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

"""
Global Default Variables
"""

# expId = 1
expId = int(sys.argv[1])
project = str(Path(__file__).absolute().parent.parent.parent)
home = str(Path.home().absolute())
datasets = home+'/Modular/Datasets/ICASSP23/'
datasets2 = home+'/Modular/Datasets/ICML23/'
datasets3 = home+'/Modular/Datasets/ICASSP24/'
datasets4 = home+'/Modular/Datasets/CAVE/chart_and_stuffed_toy_ms/'
datasets5 = home+'/Modular/Datasets/CVPR23/'
datasets6 = home+'/Modular/Datasets/KODAK/'

np.random.seed(12345)

"""
Global Subroutine Definition
"""

def completion(config, setup=None, signal=0):                                                      # stp reset fun
    """ Returns setup object for a specific config value. Depending on config value,
    re-initializes setup objet or calls reset inner method."""


    setupOpt = ss.SirenCompletion.Options({'sirenFileName': 'experiment_7',\
                                           'imagePath': config.current['imagePath'], \
                                           'maskPath': config.current['maskPath']})

    ss.SirenCompletion(opt=setupOpt)                                                        # setup

    return setup

"""
Global Experiment Options Definition
"""

expOpt = ep.Experiment.Options({'Verbose': True, 'Directory': project+'/data/'})
expVars = ep.Experiment.Variables()
expVars.hdrmap = {'sirenFileName': 'fName', 'maskPath': 'mPath'}

name = 'completionSiren'

"""
Experiments Definition
"""

if expId == 1:
    descr = 'ICLR24 KODAK 5% completion.'

    StestFileName = [datasets6+'kodim'+'{:02}'.format(val)+'r.png' for val in range(1,25)]
    MaskFileName = [datasets6 + 'kodim' + '{:02}'.format(val) + 'm5.png' for val in range(1, 25)]

    multi_run = 1                                                                                       # expOpt
    multi_step = 1
    multi_signal = 1
    varCoupling = ('StestFileName', 'MaskFileName')

elif expId == 2:
    descr = 'ICLR24 KODAK 3% completion.'

    StestFileName = [datasets6+'kodim'+'{:02}'.format(val)+'r.png' for val in range(1,25)]
    MaskFileName = [datasets6 + 'kodim' + '{:02}'.format(val) + 'm3.png' for val in range(1, 25)]

    multi_run = 1                                                                                       # expOpt
    multi_step = 1
    multi_signal = 1
    varCoupling = ('StestFileName', 'MaskFileName')

elif expId == 3:
    descr = 'ICLR24 KODAK 2% completion.'

    StestFileName = [datasets6+'kodim'+'{:02}'.format(val)+'r.png' for val in range(1,25)]
    MaskFileName = [datasets6 + 'kodim' + '{:02}'.format(val) + 'm2.png' for val in range(1, 25)]

    multi_run = 1                                                                                       # expOpt
    multi_step = 1
    multi_signal = 1
    varCoupling = ('StestFileName', 'MaskFileName')

elif expId == 4:
    descr = 'ICLR24 KODAK 1% completion.'

    StestFileName = [datasets6+'kodim'+'{:02}'.format(val)+'r.png' for val in range(1,25)]
    MaskFileName = [datasets6 + 'kodim' + '{:02}'.format(val) + 'm1.png' for val in range(1, 25)]

    multi_run = 1                                                                                       # expOpt
    multi_step = 1
    multi_signal = 1
    varCoupling = ('StestFileName', 'MaskFileName')

else:
    raise ValueError('Incorrect experiment key.')

"""
Experiment Definition & Execution
"""

expVars.update({'imagePath': StestFileName, 'maskPath': MaskFileName})                                  # expVars

def setup_reset(config, setup=None, signal=0):                                                      # stp reset fun
    return completion(config, setup=setup, signal=signal)

expOpt.update({'id': expId, 'Name': name, 'Description': descr, 'VarCoupling': varCoupling, 'multi-run': multi_run, 'saveSol': True,
               'multi-step': multi_step, 'multi-signal': multi_signal, 'MaxRunTime': 180, 'Pause': 30, 'saveSolSplit': True})
exp = ep.Experiment(expVars, setup_reset, expOpt)
exp.run()
