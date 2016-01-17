'''
Steps:
    1. Download Raw datafile
    2. Select Levels at 500,700,850
    3. Remap to lower resolution
    4. Seperate Variables
'''
import os
import sys
import tempfile
import calendar

preprocess = True 
tempdir = tempfile.mkdtemp(prefix="/gss_gpfs_scratch/vandal.t/tmp/")
tempdir = "/gss_gpfs_scratch/vandal.t/merra_2/6hour/raw/"
monthlyurl  = "ftp://goldsmr5.sci.gsfc.nasa.gov/data/s4pa/MERRA2_MONTHLY/M2IMNPASM.5.12.4/" \
            "%(year)04d/MERRA2_100.instM_3d_asm_Np.%(year)04d%(month)02d.nc4"
baseurl = "http://goldsmr5.gesdisc.eosdis.nasa.gov/data/s4pa/MERRA2/M2I6NPANA.5.12.4"\
        "/%(year)04d/%(month)02d/MERRA2_%(v)03d.inst6_3d_ana_Np." \
        "%(year)04d%(month)02d%(day)02d.nc4"
vars = ['PS', 'QV', 'SLP', 'T', 'U', 'V']
remapspecs= "~/repos/pydownscale/scripts/merra_remap_1.0x1.25"

for y in range(1980, 2016):
    for m in range(1,13):
        for d in range(1, calendar.monthrange(y,m)[1]+1):
            params = {'year': y, 'month': m, 'day': d, 'v': 100}
            if y >= 1992:
                params['v'] = 200
            if y >= 2001:
                params['v'] = 300
            if y >= 2011:
                params['v'] = 400
            fname = os.path.basename(baseurl % params)
            filepath = os.path.join(tempdir, fname)
            selfilepath = filepath.replace(".nc4","_SELLEVEL.nc4").replace("raw","levels")
            remapfilepath  = filepath.replace(".nc4","_1.0x1.25.nc4").replace("raw","remap")
            
            if (not os.path.exists(filepath)) or (m == 12 and y == 2015):
                getfilecmd =  "wget -nc -O %s %s" % (filepath,  baseurl % params)
                print getfilecmd
                os.system(getfilecmd)


'''
joinedfile = os.path.join(tempdir, "MERRA2_6Hour_instM_ASM_1980_2015.nc4")
selfiles = os.path.join(tempdir, "*SELNAME.nc4")
cdojoin = "cdo mergetime %s %s" % (selfiles, joinedfile)
os.system(cdojoin)
#os.remove(tempdir)

'''

