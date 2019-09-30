#!/usr/bin/env python

# Once installed this is not used, same as getdist script

import sys
import os

sys.path.append(os.path.realpath(os.path.dirname(__file__)))

from getdist.command_line import getdist_command

getdist_command()
