import numpy as np
import xarray as xr
import pandas as pd
from multiprocessing import Pool
import glob
from scipy import stats
import iris
import metpy.calc as mpcalc
from metpy.units import units
import datetime
import warnings
import argparse
import sys

warnings.filterwarnings("ignore")

def tb_from_olr(OLR):
    a = 1.228
    b = -1.106e-3
    sigma = 5.67e-8 # W m^-2 K^-4
    tf = (OLR/sigma)**0.25
    Tb = (-a + np.sqrt(a**2 + 4*b*tf))/(2*b)
    return Tb


def load_pp_file(tidx,var_name,sim="RAL3p2",p=slice(100,925),stream="c"):
    ds = xr.DataArray.from_iris(iris.load("/gws/ssde/j25b/kscale/USERS/bmaybee/CTC_n2560_{}/20160801T0000Z_CTC_n2560_{}_pver{}{:03d}.pp".format(
                                sim,sim,stream,tidx), var_name)[-1]).sel(latitude=slice(-5,25))
        
    if stream=="c" or stream=="d":
        ds=ds.sel(pressure=p)
    
    if phase=="ocean":
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby("longitude").sel(longitude=slice(-40,-5))
        ds = ds.interp(longitude=np.arange(-40,-5,0.1),latitude=np.arange(0,25,0.1))
    else:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby("longitude").sel(longitude=slice(-25,35))
        ds = ds.interp(longitude=np.arange(-25,35,0.1),latitude=np.arange(-5,25,0.1))
    return ds.compute()

    

def composite_field(field,clat,clon,ll_border=7.5,size=31,pad=0.5,err=True):
    #relies on data with coordinates in form (pressure,latitude,longitude), where lat and lon are both ascending.
    s=len(field.shape)
    if s==2:
        field=field.expand_dims("pressure",axis=0).expand_dims("scale",axis=0)
    elif s==3:
        field=field.expand_dims("scale",axis=0)
    
    field_comp=field.sel(latitude=slice(clat-ll_border-pad,clat+ll_border),
                         longitude=slice(clon-ll_border-pad,clon+ll_border))[:,:,:size,:size]

    if field_comp.shape[-1] != size or field_comp.shape[-2] != size and err==True:
        print("Composite sampling error, ",field_comp.shape,clat,clon)
        return
    
    if s==2:
        field_comp=field_comp.isel(pressure=0,scale=0)
    elif s==3:
        field_comp=field_comp.isel(scale=0)
    return field_comp

