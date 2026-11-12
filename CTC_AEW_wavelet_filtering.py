from utils import *
import wclass
import constants as cnst

# wclass and constants underping wavelet filtering; code available at https://github.com/cornkle/proj_CEH/tree/master/land_wavelet

sim=sys.argv[1]

dataAll=xr.merge([xr.open_dataset(f"/gws/ssde/j25b/lmcs/bmaybee/aew_work/DY2_CTC_{sim}_dynamics_0p5deg.nc"),
                  xr.open_dataset(f"/gws/ssde/j25b/lmcs/bmaybee/aew_work/DY2_CTC_{sim}_thermo_0p5deg.nc"),
                  xr.open_dataset(f"/gws/ssde/j25b/lmcs/bmaybee/aew_work/DY2_CTC_{sim}_PV_0p5deg.nc").isel(pressure=slice(None,None,-1))]).sel(
                latitude=slice(0,25),longitude=slice(-50,30),pressure=slice(None,950))
wObj = wclass.landwav('kscale_tseries')
scales = wObj.scales
# factor of 8 to account for 3-hourly vertical data. 2 and 8 set frequencies to 2 and 8 day bounds respectively
aew_scales = np.where((scales>8*2) & (scales<8*8))[0]

def aew_filter(signal):
    wObj = wclass.landwav('kscale_tseries')
    wObj.read_img(np.expand_dims(signal,1)[:,:None], np.arange(len(signal)), np.arange(len(signal)))
    coeffs, power, scales, period = wObj.applyWavelet(normed='none')
    variableIndi, scalesIndi = wObj.applyInverseWavelet(per_scale=True)
    # filtering applied in reconstruction: only include frequency scales set above
    return np.mean(signal)+np.sum(variableIndi[aew_scales,:,0],axis=0)


def plev_filter(plev):
    ds=data.sel(pressure=plev)
    print(plev)
    for i in range(ds.shape[1]):
        for j in range(ds.shape[2]):
            ds[:,i,j]=aew_filter(ds[:,i,j].values)
    
    return ds

final=[]
for var in ["x_wind","y_wind","pressure_tendency","air_temperature","specific_humidity","potential_vorticity"]:
    print(var)
    data=dataAll[var]
    p=Pool(len(data.pressure))
    out=p.map(plev_filter,data.pressure.values)
    out=xr.concat(out,dim="pressure")
    final.append(out.sortby("pressure",ascending=False))
        
xr.merge(final).to_netcdf(f"/gws/ssde/j25b/lmcs/bmaybee/aew_work/DY2_CTC_{sim}_0p5deg_3-5d_filter_EAtl.nc")