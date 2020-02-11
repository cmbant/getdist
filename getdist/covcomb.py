# usage:
# python covcmb.py out.covmat in1.covmat in2.covmat
# Nb. in1 values take priority over in2

import sys
from getdist import covmat

if len(sys.argv) < 3:
    print('Usage: python covcmb.py out.covmat in1.covmat in2.covmat [in3.covmat...]')
    sys.exit()

foutname = sys.argv[1]

cov = covmat.CovMat(sys.argv[2])

for fname in sys.argv[3:]:
    print('merging: ' + fname)
    cov = cov.mergeCovmatWhereNew(covmat.CovMat(fname))

cov.saveToFile(foutname)
