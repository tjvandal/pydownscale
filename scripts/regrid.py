import os
import sys
import subprocess

remapfile = os.path.join(os.path.dirname(__file__), "merra_remap_1.0x1.25")
basencdir = "/gss_gpfs_scratch/vandal.t/cmip5/historical/atm/day/ccsm4/"
rawdir = os.path.join(basencdir, "raw")
remapdir = os.path.join(basencdir, "remap")

subdirs = os.listdir(rawdir)

for d in subdirs:
    print "subdirectory", d
    subpath = os.path.join(rawdir, d)
    varpath = os.path.join(remapdir, d)
    if not os.path.exists(varpath):
        os.mkdir(varpath)
    if not os.path.isdir(subpath):
        continue
    for f in os.listdir(subpath):
        if f[-3:] in (".nc", "nc4"):
            rawpath = os.path.join(subpath, f)
            remappath = os.path.join(varpath, f)
            args = ['cdo', 'remapbil,%s' % remapfile, rawpath, remappath]
            print " ".join(args)
            if not os.path.exists(remappath):
                subprocess.check_call(args)
