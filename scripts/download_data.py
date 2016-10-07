__author__ = 'tj'
import os
import sys

BASE_DIR = "/gss_gpfs_scratch/vandal.t/"

def mergetime(dir, var):
    files = [os.path.join(dir, f) for f in os.listdir(dir)]
    cmd = "cdo mergetime %s/*.nc %s_1948_2014.nc" % (dir, var)
    os.chdir(dir)
    to_file = "%s_1948_2014.nc" % var
    if os.path.exists(to_file):
        return
    print "run command:", cmd
    os.system(cmd)
    for f in files:
        print "remove?", f
        #os.remove(f)

def checkpath(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def download_ncep_daily():
    years = range(1948, 2015)
    variables_levels = ['air', 'rhum', 'uwnd', 'vwnd']
    ncep_pressure_levels = "ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/pressure/%s.%i.nc"
    cmd = 'wget %s' % (ncep_pressure_levels)
    ncep_dir = os.path.join(BASE_DIR, "ncep/daily/downscale-data")
    plv_path = ncep_dir #os.path.join(ncep_dir, 'plv')

    checkpath(ncep_dir)
    checkpath(plv_path)

    for var in variables_levels:
        varpath = os.path.join(plv_path, var)
        checkpath(varpath)
        os.chdir(varpath)

        for y in years:
            fname = (ncep_pressure_levels % (var, y)).split("/")[-1]
            if not os.path.exists(fname):
                os.system(cmd % (var, y))

        #mergetime(varpath, var)

    surface_url = 'ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface/pr_wtr.eatm.%i.nc'
    cmd = 'wget %s' % surface_url
    varpath = os.path.join(ncep_dir, 'pr_wtr')
    checkpath(varpath)
    os.chdir(varpath)
    for y in years:
        fname = (surface_url % y).split("/")[-1]
        if not os.path.exists(fname):
            os.system(cmd % y)

    #mergetime(varpath, 'pr_wtr')

    slp_url = 'ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface/slp.%i.nc'
    cmd = 'wget %s' % slp_url
    varpath = os.path.join(ncep_dir, 'slp')
    checkpath(varpath)
    os.chdir(varpath)
    for y in years:
        fname = (slp_url % y).split("/")[-1]
        if not os.path.exists(fname):
            os.system(cmd % y)

   # mergetime(varpath, 'slp')


def get_ncep_file_paths():
    files = []
    dir = os.path.join(BASE_DIR, "ncep/daily")
    for (dirpath, dirnames, filenames) in os.walk(dir):
        for f in filenames:
            if '1948_2014' in f:
                files.append(os.path.join(dirpath, f))

    return files


if __name__ == "__main__":
    download_ncep_daily()

    #files = get_ncep_file_paths()
    #print files
    #import xarray
    #d = xarray.open_mfdataset(files)
    #print d['rhum'].values

