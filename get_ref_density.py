from utils import *

sim=sys.argv[1]

period=pd.date_range("2016-08-01","2016-09-09")

out=[]
for date in period:
    print(date)
    files=glob.glob(f"/gws/ssde/j25b/kscale/DATA/outdir_20160801T0000Z/DMn1280GAL9/channel_n2560_{sim}/profile_*/2016{date.month:02d}{date.day:02d}*")
    temp=[]
    for ffile in files:
        ds=xr.open_dataset(ffile).sel(latitude=slice(0,25), longitude=slice(-50, 30)
                                 )[["air_temperature","specific_humidity"]]
        temp.append(ds)
    temp=xr.concat(temp,dim="pressure").sortby("pressure")

    mr=mpcalc.mixing_ratio_from_specific_humidity(temp.specific_humidity)
    density=mpcalc.density(temp.pressure.values[:,None,None,None] * units("hPa"), temp.air_temperature * units("K"), mr)
    out.append(density.metpy.dequantify())
    
out=xr.concat(out,dim="time")
print(out)
out.mean(dim="time").to_netcdf(f"/gws/ssde/j25b/lmcs/bmaybee/aew_work/DY2_CTC_{sim}_EAtl_density_0p5deg.nc")
