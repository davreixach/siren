#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SIREN package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
sirenSetup Class File
=================================

Classes for helping in the implementation of an experimental setup to test Siren algorithm on
tensors. They include the following problems:

    - Tensor Completion via Siren algorithm

"""

from __future__ import print_function

from builtins import range
from builtins import object
from builtins import eval
from collections import namedtuple

import os
import copy
import numpy as np
# import torch
# import pandas as pd
import cv2

import emics.util as emu
from emics import experiment

# from tensor import lrd
# from tensor import ccmod
# from tensor import cbpdn

# from sporco.admm import admm
# from sporco import cnvrep
# from sporco.admm import ccmod as spccmod
# from sporco.dictlrn import dictlrn
# from sporco.util import u

# from scipy.spatial.distance import cdist, squareform

import make_figures

__author__ = """David Reixach <david.reixach@upc.edu>"""

NOTHING = object()


class SirenCompletion(experiment.GenericSetup):
    """
    Generic class for performing training with lrd.
    """

    # class Parameters(dict):
    #     """Parameters to store results contents."""
    #
    #     # defaults = {'Rank_': "opt['Rank_']", 'lmbda_': "opt['lambda_']", 'mu_': "opt['mu_']", 'Rank': "opt['Rank']",
    #     #             'lmbda': "opt['lambda']", 'mu': "opt['mu']", 'n_x': "opt['opt_k_']['MaxMainIter']",
    #     #             'n_alt': "opt['opt_x']['MaxMainIter']", 'n_d': "opt['opt_d']['MaxMainIter']"}
    #     defaults = {}
    #
    #     def __init__(self, par=None):
    #         """Init Options Object."""
    #
    #         if par is None:
    #             par = {}
    #
    #         self.update(__class__.defaults)                                                         # set defaults
    #         self.update(par)


    class Options(experiment.GenericSetup.Options):
        """General dictionary learning algorithm options.

        Options:

          ``Verbose`` : Flag determining whether iteration status is
          displayed.

          ``StatusHeader`` : Flag determining whether status header and
          separator are displayed.

          ``IterTimer`` : Label of the timer to use for iteration times.

          ``MaxMainIter`` : Maximum main iterations.

          ``Callback`` : Callback function to be called at the end of
          every iteration.
        """

        # opt_k_ = lrd.LRD1modeOptions({'MaxMainIter': 50, 'theta': None, 'gamma': 0.0}, lmbda=0.0)  # code opt
        # opt_x = lrd.LRD.Options({'MaxMainIter': 60})

        # opt_dl = dictlrn.DictLearn.Options({'MaxMainIter': 60})                                   # dict opt
        # opt_d = spccmod.ConvCnstrMODOptions({'MaxMainIter': 1}, method='ism')
        # opt_d = ccmod.ConvCnstrMOD_IterSM.Options({'MaxMainIter': 1})
        # opt_d = ccmod.CCMOD_MaskedGPU.Options({'MaxMainIter': 1})

        # print(isinstance(opt_d, ccmod.ConvCnstrMOD_IterSM.Options))

        # isc_flds = {'isfld_g': ['Iter', 'Time'], 'hdrtxt_g': ['Itn'], 'hdrmap_g': {'Itn': 'Iter'},
        #             'isfld_l': ['Iter', 'ObjFun', 'DFid', 'RegL1', 'RegL2', 'PrimalRsdl', 'DualRsdl', 'Rho', 'Rsdl', 'L', 'RegTV'],
        #             'hdrtxt_l': ['Fnc', 'r', 's', u('ρ'), 'DFid', u('Regℓ2'), 'Rsdl', 'L', 'RegTV'],
        #             'hdrmap_l': {'Fnc': 'ObjFun', 'r': 'PrimalRsdl', 's': 'DualRsdl', u('ρ'): 'Rho', 'DFid': 'DFid',
        #                          u('Regℓ2'): 'RegL2', 'Rsdl': 'Rsdl', 'L': 'L', 'RegTV': 'RegTV'}}
        #
        # isc_flds_d = {'isfld_g': ['Iter', 'ObjFunD', 'DPrRsdl', 'DDlRsdl', 'DRho', 'Time'],
        #               'isfld_l': ['ObjFunX', 'XPrRsdl', 'XDlRsdl', 'XRho'], 'isxmap': {'ObjFunX': 'ObjFun',
        #               'XPrRsdl': 'PrimalRsdl', 'XDlRsdl': 'DualRsdl', 'XRho': 'Rho'}, 'isdmap': {'ObjFunD': 'DFid',
        #                 'DPrRsdl': 'PrimalRsdl', 'DDlRsdl': 'DualRsdl', 'DRho': 'Rho'}, 'hdrmap': {'Itn': 'Iter',
        #                 'FncX': 'ObjFunX', 'r_X': 'XPrRsdl', 's_X': 'XDlRsdl', u('ρ_X'): 'XRho', 'FncD': 'ObjFunD',
        #                 'r_D': 'DPrRsdl', 's_D': 'DDlRsdl', u('ρ_D'): 'DRho'}}

        defaults = copy.deepcopy(experiment.GenericSetup.Options.defaults)

        defaults.update({'sirenFileName': 'experiment_0', 'imagePath': '', 'maskPath': ''})

        # defaults.update({'MaxMainIter': 60, 'PercentMissing': 30})

        # print(isinstance(defaults['opt_d'], ccmod.ConvCnstrMOD_IterSM.Options))

        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              DictLearn algorithm options
            """

            experiment.GenericSetup.Options.__init__(self, opt)                                     # super init.
            # self.update({'initDict': self['initDict'].lower()})


    def __init__(self, opt=None, res_par=None):

        self.Init = False

        if opt is None:                                                                             # default opt
            opt = SirenCompletion.Options()

        self.opt = opt                                                                              # init vars.
        # common_keys = ['HighMemSolve']                                                              # common key
        # self.opt_auto_inheritance(common_keys=common_keys)
        #
        # self.Sinput = S                                                                             # save inputs
        # self.Dinput = D
        #
        # # self.homo_signal = True                                                                 # homogeneous signal
        # self.solveDict = False
        #
        # self.Mask = None                                                                            # not used
        # # self.Stest = None

        self.res_par = None                                                                         # set in reset()

        # self.multi_signal = len(S)                                                                  # S is list

        # if len(S.shape) == 1 and S[0].size > 1:                                                     # signal is not h.
        #     self.homo_signal = False

        # self.init_aux_objects()                                                                     # aux obj.
        # self.init_code_solver(init=False)

        # if self.solveDict:                                                                          # solveDict
        #     d0 = self.init_dictionary()                                                             # Init Dict.
        #     self.D = cnvrep.Pcn(d0, d0.shape, self.cri.Nv, dimN=self.opt['dimN'],                   # Normalize dict.
        #                         dimC=self.cri.dimC, crp=True, zm=self.opt['opt_d']['ZeroMean'])
        #     self.init_dict_solver()

        self.timer = emu.Timer('execution')

        self.reset(res_par=res_par)                                                                 # reset

        self.Init = True

    def reset(self, opt=NOTHING, res_par=NOTHING):
        """Reset parameters."""

        # update_aux_objects = False                                                                       # multi-signal

        if opt is self.opt and self.Init:                                                           # opt update
            opt = NOTHING

        # if res_par is self.res_par and self.Init:
        #     res_par = NOTHING

        if opt is not NOTHING:                                                                      # opt reset
            if opt is None:
                opt = SirenCompletion.Options()
            # if opt['signal'] != self.opt['signal'] or \
            #         opt['PercentMissing'] != self.opt['PercentMissing'] or \
            #         opt['shpD'] != self.opt['shpD']:
            # update_aux_objects = True

            self.opt = opt
            # common_keys = ['HighMemSolve']                                                          # common key
            # self.opt_auto_inheritance(common_keys=common_keys)

        # if res_par is not NOTHING:                                                                  # res_par reset
        #     if res_par is None:
        #         res_par = TensorReconstruction.Parameters({'M': "cri.M"})
        #     self.res_par = res_par
        #
        # if update_aux_objects or not self.Init:
        #     print('\nUpdating aux objects...\n')
        #     self.init_aux_objects()                                                                 # aux objects
        #     init_x_solver = True
        # else:
        #     init_x_solver = False
        #
        # if self.solveDict:                                                                          # solveDict
        #     d0 = self.init_dictionary()                                                             # Init Dict.
        #     self.D = cnvrep.Pcn(d0, d0.shape, self.cri.Nv, dimN=self.opt['dimN'],                   # Normalize dict.
        #                         dimC=self.cri.dimC, crp=True, zm=self.opt['opt_d']['ZeroMean'])
        #
        # self.init_code_solver(init_solver=init_x_solver)                                            # init code solver
        #
        # if self.solveDict:
        #     self.init_dict_solver()                                                                 # init dict solver

        # self.set_res_parameters()                                                                   # opt dependant
        self.clear_stat()                                                                           # clear stat

        # self.xstep.reset()                                                                          # reset xstep

    def run(self):
        """Run."""

        sirenFileName = self.opt['sirenFileName']
        imagePath = self.opt['imagePath']
        maskPath = self.opt['maskPath']

        self.timer.reset(['execution'])
        self.timer.start(['execution'])

        os.system("rm -rf ./logs/" + sirenFileName + "/")                               # remove old data

        os.system("python3 experiment_scripts/train_img_inpainting.py --experiment_name=" + sirenFileName +\
                    "--dataset='custom' --custom_image=" + imagePath + "--mask_path=" + maskPath +\
                  "--num-epochs=" + " %.i"% int(1000))

        self.timer.stop(['execution'])

        self.result = self.catch_results(self.elapsed('execution'))

        if self.opt['Verbose']:
            print("Siren solve time: %.2fs" % self.elapsed('execution'), "\n")
            print("Test "+emu.ntpl2string(self.result) + "\n")

    def catch_results(self, time):
        """Catch siren completion results"""

        summaryPath = "./logs/" + self.opt['sirenFileName'] + "/summaries/"

        make_figures.extract_image_psnrs(dict({'base': summaryPath}))
        # make_figures.extract_image_times(dict({'base': summaryPath}))

        arr_psnrs = np.load(summaryPath + "psnrs.npy")
        # arr_times = np.load(summaryPath + "times.npy")

        resTuple = collections.namedtuple('Result', ['PSNR', 'Time'])                   # as namedtuple
        return resTuple(arr_psnrs[-1], time)

    def catch_solutions(self):
        """Catch siren completion solutions"""

        summaryPath = "./logs/" + self.opt['sirenFileName'] + "/summaries/"

        make_figures.extract_images_from_summary(summaryPath, 'train_pred_img', suffix='',\
                                                 img_outdir='./out/', colormap=None)

        R_nl = emu.nestedList(())                                                                   # nestedList
        S_nl = emu.nestedList(())
        Stest_nl = emu.nestedList(())
        Mask_nl = emu.nestedList(())

        Stest = cv2.imread(self.opt['imagePath'])
        Mask = cv2.imread(self.opt['maskPath'])

        R_nl.setElement(cv2.imread(summaryPath + "0009.png"), id=None)                              # get
        S_nl.setElement(Stest*Mask, id=None)
        Stest_nl.setElement(Stest, id=None)
        Mask_nl.setElement(Mask, id=None)

        save_list = [R_nl, S_nl, Stest_nl, Mask_nl]                                                 # as dict
        keys_list = ['R', 'S', 'Stest', 'Mask']

        return dict(zip(keys_list, save_list))

    def set_res_parameters(self):
        """Set results parameters list."""

        self.parameters = dict(zip(self.res_par.keys(), [eval(val, self.__dict__) for val in self.res_par.values()]))

    def get_full_result(self, results):
        """
        Create results tuple. As fields it includes user specified parameters and inner object results.
        """

        resTuple = namedtuple('Results', list(self.parameters.keys()) + list(results._fields))          # cast to ntpl
        return resTuple(*self.parameters.values(), *results)

    def get_solutions(self):
        """Get inner objects solutions."""

        return self.catch_solutions()

    def get_results(self):
        """Get xstep objects results"""

        results_nl = emu.nestedList(())                                                                 # nestedList
        results = self.get_full_result(self.result)                                                     # full result
        results_nl.setElement(emu.ntpl2array(results), id=None)                                         # append

        return {'Siren': results_nl}

    def get_stats(self):
        """Get d-object and xstep object statistics"""

        return {}

    def save_solutions(self, file_name, id=None):
        """Save collected solutions."""

        solutions = self.get_solutions()                                                                # get

        for el in solutions.items():                                                                    # save
            el[1].save(file_name, key=el[0], id=id)

    def save_results(self, file_name, id=None):
        """Save collected results."""

        results = self.get_results()                                                                    # get

        results['Siren'].save(file_name, key='Siren', id=id)                                            # save

    def save_stats(self, file_name, id=None):
        """Save collected statistics."""

        pass

    def clear_stat(self):
        """Clear solvers iteration stat. objects."""

        pass

    def save_solutions_split(self, file_name, id=None):
        """Save collected solutions split for each configuration."""

        solutions = self.get_solutions()                                                                # get

        id2 = list(id)
        id2[0] = 0
        id2 = tuple(id2)

        for el in solutions.items():                                                                    # save
            el[1].save(file_name, key=el[0], id=id2)