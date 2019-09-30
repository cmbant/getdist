#!/usr/bin/env python

# Once installed this is not used, same as getdist script

from getdist.command_line import getdist_command
import sys
import os

sys.path.append(os.path.realpath(os.path.dirname(__file__)))

getdist_command()
