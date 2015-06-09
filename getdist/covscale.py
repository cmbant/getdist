from __future__ import print_function
import sys
import fnmatch
import os
from getdist import covmat

if len(sys.argv) < 4:
    print('covscale rescales parameter(s) in all .covmat files in a directory and outputs to another directory')
    print('Usage: python covscale.py in_dir out_dir param1:param2:.. fac1:fac2:..')
    sys.exit()

indir = os.path.abspath(sys.argv[1]) + os.sep
outdir = os.path.abspath(sys.argv[2]) + os.sep
pars = sys.argv[3].split(':')
factors = sys.argv[4].split(':')

if not os.path.exists(outdir): os.makedirs(outdir)

for f in os.listdir(indir):
    if fnmatch.fnmatch(f, "*.covmat"):
        print(indir + f)
        cov = covmat.CovMat(indir + f)
        for par, factor in zip(pars, factors):
            cov.rescaleParameter(par, float(factor))
        cov.saveToFile(outdir + f)
