#!/shared/apps/sage/sage-5.12/spkg/bin/sage -python
'''
mpirun -np 1 /shared/apps/sage/sage-5.12/spkg/bin/sage -python ~/repos/pydownscale/scripts/merra_preprocess.py
'''

import os
import sys
sys.path.insert(0, "/home/vandal.t/anaconda/lib/python2.7/site-packages")

import subprocess
from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

base_dir = "/gss_gpfs_scratch/vandal.t/merra_2/"
raw_dir = os.path.join(base_dir, "6hour", "raw")
remap_dir = os.path.join(base_dir, "6hour", "remap")
surface_raw_dir = os.path.join(base_dir, "6hour", "M2I1NXASM.5.12.4")


remap = "/home/vandal.t/repos/pydownscale/scripts/merra_remap_1.0x1.25"
vars = ['T', 'U', 'V', 'H', 'QV', 'SLP', 'PS'] 
surface_vars = ['T2M', 'TS', 'V2M', 'U2M', 'TQV', 'TQL'] 
months = range(1,13)
years = range(1980, 2016)

files = None
if rank == 0:
    files =sorted([(os.path.join(raw_dir, f), f) for f in os.listdir(raw_dir) if f[-3:] == 'nc4'])
    files = numpy.array_split(files, size)

files = comm.scatter(files, root=0)
for rawfile, f in files:
    # remap file to a higher resolution if needed
    remapfile = os.path.join(remap_dir, f.replace(".nc4", "_1.0x1.25.nc4"))
    if not os.path.exists(remapfile):
        args = ['cdo', 'remapbil,%s' % remap, rawfile, remapfile]
        subprocess.check_call(args)


    levelfile = 'lev_%s' % f
    for var in vars:
        varpath = os.path.join(base_dir, "6hour", var)
        varfile = os.path.join(varpath, f.replace(".nc4", "_1.0x1.25_%s.nc4" % var))
        if not os.path.exists(varfile):
            if not os.path.exists(varpath):
                os.mkdir(varpath)

            # select levels and variables, put into directories
            if not os.path.exists(levelfile):
                argslevel = ['cdo', 'sellevel,500,700,850', remapfile, levelfile]
                try:
                    subprocess.check_call(argslevel)
                except Exception as err:
                    print err
            #       sys.exit()
            #        os.remove(remapfile)

            if var in ['T', 'U', 'V', 'QV', 'H']:
                argsvar = ['cdo', 'selname,%s' % var , levelfile, varfile] 
            else:
                argsvar = ['cdo', 'selname,%s' % var , remapfile, varfile]

            try:
                subprocess.check_call(argsvar)
            except:
                os.remove(remapfile)


        # lets compute daily means
        daypath = os.path.join(base_dir, "daily", var)
        dayfile = os.path.join(daypath, f.replace(".nc4", "_1.0x1.25_%s.nc4" % var)) 
        if not os.path.exists(dayfile):
            if not os.path.exists(daypath):
                os.mkdir(daypath)
            args = ['cdo', 'daymean', varfile, dayfile]
            try:
                subprocess.check_call(args)
            except:
                os.remove(varfile)
    
    if os.path.exists(levelfile):
        os.remove(levelfile)


if rank == 0:
    years = numpy.array_split(years, size)
else:
    years = None

years = comm.scatter(years, size)

if rank == 0:
    for var in vars:
        yearpath = os.path.join(base_dir, "daily_merged", var)
        if not os.path.exists(yearpath):
            os.mkdir(yearpath)

# merge daily files into a year
for y in years:
    for var in vars:
        daypath = os.path.join(base_dir, "daily", var)
        yearpath = os.path.join(base_dir, "daily_merged", var)
        yearfile = os.path.join(yearpath, "MERRA2.instD_3d_ana_Np.%04d_1.0x1.25_%s.nc4" % (y, var))
        if (os.path.exists(yearfile)): # and (os.path.getsize(yearfile) != 0):
 #           os.remove(yearfile) 
            continue

        argsmerge = ['cdo', 'mergetime', os.path.join(daypath, "*Np.%04d*" % y), yearfile]
        try:
            subprocess.check_call(argsmerge)
        except Exception as err:
            print err
            if os.path.exists(yearfile):
                os.remove(yearfile)
        #cmd = " ".join(argsmerge)
        #os.system(cmd)

## lets work on the surface directory
if not os.path.exists(surface_raw_dir):
    sys.exit()

if rank == 0:
    surface_files = []
    for root, dirs, files in os.walk(surface_raw_dir):
        surface_files += [[os.path.join(root, f), f] for f in files if f[-3:] == 'nc4']
    surface_files = numpy.array_split(surface_files, size)
else:
    surface_files = None

surface_files = comm.scatter(surface_files)

for fpath, f in surface_files:
    remapdir = os.path.join(base_dir, "6hour", "remap_surface")
    remapfile = os.path.join(remapdir, f.replace(".nc4", "_1.0x1.25.nc4"))
    if not os.path.exists(remapfile):
        tempfile = "temp" + f 
        argsvars = ['cdo', 'selname,%s' % ",".join(surface_vars), fpath, tempfile]
        subprocess.check_call(argsvars)
        args = ['cdo', 'remapbil,%s' % remap, tempfile, remapfile]
        subprocess.check_call(args)
        os.remove(tempfile)

    for v in surface_vars:
        vardir = os.path.join(base_dir, "6hour", v)
        varfile = os.path.join(vardir, os.path.basename(remapfile).replace(".nc4", "_%s.nc4" % v))
        if not os.path.exists(varfile):
            if not os.path.exists(vardir):
                os.mkdir(vardir)
            argsvar = ['cdo', 'selname,%s' % v , remapfile, varfile] 
            try:
                subprocess.check_call(argsvar)
            except Exception as err:
                print err
                os.remove(remapfile)
                continue
            dailyvardir = os.path.join(base_dir, "daily", v)
            dailyvarfile = os.path.join(dailyvardir, os.path.basename(varfile))
            if not os.path.exists(dailyvarfile):
                if not os.path.exists(dailyvardir):
                    os.mkdir(dailyvardir)
                argsmean = ['cdo', 'daymean', varfile, dailyvarfile]
                try:
                    subprocess.check_call(argsmean)
                except Exception as err:
                    print err
                    os.remove(varfile)
                    continue

for y in years:
    for var in surface_vars:
        daypath = os.path.join(base_dir, "daily", var)
        yearpath = os.path.join(base_dir, "daily_merged", var)
        yearfile = os.path.join(yearpath, "MERRA2.instD_3d_ana_Nx.%04d_1.0x1.25_%s.nc4" % (y, var))
        if not os.path.exists(yearpath):
            os.mkdir(yearpath)
        if (os.path.exists(yearfile)): # and (os.path.getsize(yearfile) != 0):
#            os.remove(yearfile) 
            continue

        argsmerge = ['cdo', 'mergetime', os.path.join(daypath, "*Nx.%04d*" % y), yearfile]
        try:
            subprocess.check_call(argsmerge)
        except Exception as err:
            print err
            if os.path.exists(yearfile):
                os.remove(yearfile)
