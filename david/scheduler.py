#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
ICLR2025 Experimentation Scheduler Script
=========================================

"""

import sys
import os
from pathlib import Path

"""
File Management Variables
"""
scripts = str(Path(__file__).parent)
main = scripts+'/completionSirenExp.py'
# main = scripts+'/denoisingExp2.py'

"""
Run experiments
"""
for id,exp in enumerate(sys.argv[1:]):
    os.system("python3 "+main+" %.0i"% int(exp))
