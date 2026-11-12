from utils import *


# these are L/2
scales=[125000.,  250000., 500000., 1000000.]

##################################################################################################
parser=argparse.ArgumentParser()
parser.add_argument("-f", "--fields")
parser.add_argument("-s", "--sim", required=False, default='RAL3p2')
parser.add_argument("-p", "--phase", required=False, default="growth")
args = parser.parse_args()

fields=args.fields
sim=args.sim
phase=args.phase
##################################################################################################

aew_tracks=xr.open_dataset(f"~/QTrack/CTC_{sim}_AEW_tracks_post_processed.nc")
if phase=="growth" or phase=="decay" or phase=="prop":
    rate=aew_tracks.AEW_strength.rolling(time=4).mean().differentiate("time",datetime_unit="D")
    if phase=="growth":
        aew_tracks=aew_tracks.where(rate>2.5e-6, drop=True) # == 1e-5/s/d since 6 hourly data
    elif phase=="prop":
        aew_tracks=aew_tracks.where(rate<2.5e-6, drop=True)     
    else:
        aew_tracks=aew_tracks.where(rate<-2.5e-6, drop=True) # not used for any paper analysis
    aew_tracks=aew_tracks.where((aew_tracks.AEW_lat_smooth>5) & (aew_tracks.AEW_lat_smooth<15) & 
                                (aew_tracks.AEW_lon_smooth>-15) & (aew_tracks.AEW_lon_smooth<25), drop=True)

elif phase=="ocean":
    # includes all growth phases
    aew_tracks=aew_tracks.where((aew_tracks.AEW_lat_smooth>7.5) & (aew_tracks.AEW_lat_smooth<17.5) & 
                                (aew_tracks.AEW_lon_smooth>-32) & (aew_tracks.AEW_lon_smooth<-15), drop=True)


    

def isetQ_composites(idx):
    # only use 0p5deg ISET file to get pressure levels for Qprof (bit silly design!)
    iset=xr.open_dataset(f"/gws/ssde/j25b/kscale/LoSSETT_data/DYAMOND_Summer/global_n1280_GAL9/channel_n2560_{sim}/0p5deg"+
                     f"/channel_n2560_{sim}.n1280GAL9_0p5deg_inter_scale_transfer_of_kinetic_energy_Lmin_00055_Lmax_01705_20160801.nc")
    Qprofs=xr.open_dataset(f"/gws/ssde/j25b/lmcs/bmaybee/aew_work/Channel_{sim}_ea-waf_Q_diagnostics_allZ_plevs.nc").sel(latitude=slice(-5,25))
    Qprofs=Qprofs.assign_coords(level=iset.pressure.values[:-1][::-1]).rename({"level":"pressure"}).transpose("time","pressure","latitude","longitude")
    
    var_list=["Dl_u","Q_tot","Q1","Q2","Q3"]
    comp_dict={}
    for var in var_list:
        comp_dict[var]=[]

    period_seg=aew_tracks.time.values[idx*fact:(idx+1)*fact]

    for t in period_seg:
        print(t)
        tstampL=pd.Timestamp(aew_tracks.time.sel(time=t).values)-pd.Timedelta(6,"h")

        # Have to load n640 LoSSETT output by pressure level
        try:
            iset=isetT.sel(time=t)
        except:
            isetT=[]
            for ffile in glob.glob(f"/gws/ssde/j25b/kscale/LoSSETT_data/DYAMOND_Summer/global_n1280_GAL9/channel_n2560_{sim}/n640/channel_n2560_{sim
                                }.n1280_GAL9_inter_scale_transfer_of_kinetic_energy_Lmin_00032_Lmax_02500_2016%02d%02dT%02d_p*_n1280_embedded_n640.nc"%(
                                tstampL.month,tstampL.day,int(tstampL.hour/12)*12)):
                ds=xr.open_dataset(ffile).sel(longitude=slice(-40,35),latitude=slice(-5,25),length_scale=scales)
                isetT.append(ds.interp(latitude=np.arange(-5,25.1,0.2),longitude=np.arange(-40,35.1,0.2)))
            isetT=xr.concat(isetT,dim="pressure")
        iset=isetT.sel(time=t)
        
        for s in aew_tracks.system.values:
            aew=aew_tracks.sel(time=t,system=s)
            if np.isnan(float(aew.AEW_strength)):
                continue

            try:
                comp=composite_field(iset.Dl_u,aew.AEW_lat_smooth,aew.AEW_lon_smooth,ll_border=7.4,size=75)
                comp_dict["Dl_u"].append(comp.values)
            except:
                continue
    
            for var in var_list[1:]:
                try:
                    comp=composite_field(Qprofs.sel(time=t)[var],aew.AEW_lat_smooth,aew.AEW_lon_smooth,size=30)
                    #print(var,comp.shape)
                    comp_dict[var].append(comp.values)
                except:
                    continue
    
    N=len(comp_dict[var])
    comp_dict["Dl_u_mean"]=xr.DataArray(np.nanmean(comp_dict["Dl_u"],axis=0),name="Dl_u_mean",dims=["length_scale","pressure","lat","lon"],
                                        coords=[iset.length_scale.values/1000,iset.pressure.values,np.arange(-7.4,7.6,0.2),np.arange(-7.4,7.6,0.2)])
    comp_dict["Dl_u_std"]=xr.DataArray(np.nanstd(comp_dict["Dl_u"],axis=0),name="Dl_u_std",dims=["length_scale","pressure","lat","lon"],
                                        coords=[iset.length_scale.values/1000,iset.pressure.values,np.arange(-7.4,7.6,0.2),np.arange(-7.4,7.6,0.2)])
    del comp_dict["Dl_u"]
    
    for var in var_list[1:]:
        comp_dict[var+"_mean"]=xr.DataArray(np.nanmean(comp_dict[var],axis=0),name=var+"_mean",dims=["pressure","latitude","longitude"],
                                            coords=[Qprofs.pressure.values[::-1],np.arange(-7.5,7.5,0.5),np.arange(-7.5,7.5,0.5)])
        comp_dict[var+"_std"]=xr.DataArray(np.nanstd(comp_dict[var],axis=0),name=var+"_std",dims=["pressure","latitude","longitude"],
                                           coords=[Qprofs.pressure.values[::-1],np.arange(-7.5,7.5,0.5),np.arange(-7.5,7.5,0.5)])
        del comp_dict[var]
    
    out=xr.merge([comp_dict[field] for field in list(comp_dict.keys())])
    out.attrs["number_cores"]=N
    return out


def dynamics_composites():    
    comp_dict={}
    for var in var_list:
        comp_dict[var]=[]

    period=aew_tracks.time.values
    print(dyn)
    for t in period:
        print(t)
        
        data_dict={}
        data_dict["x_wind"]=dyn["x_wind"].sel(time=t)
        data_dict["y_wind"]=dyn["y_wind"].sel(time=t)
        data_dict["horizontal_divergence"]=dyn["horizontal_divergence"].sel(time=t)
        data_dict["vertical_vorticity"]=dyn["vertical_vorticity"].sel(time=t)
        data_dict["pv"]=pv.sel(time=t)
        data_dict["pv_anom"]=data_dict["pv"] - pv_clim.sel(hour=pd.Timestamp(t).hour).values
        
        for s in aew_tracks.system.values:
            aew=aew_tracks.sel(time=t,system=s)
            if np.isnan(float(aew.AEW_strength)):
                continue
            if aew.AEW_lat_smooth < 7.5:
                continue
    
            for var in var_list:
                #try:
                comp_dict[var].append(composite_field(data_dict[var],aew.AEW_lat_smooth,aew.AEW_lon_smooth).values)
                #except:
                #    continue

    N=len(comp_dict[var])
    for var in var_list:
        comp_dict[var+"_mean"]=xr.DataArray(np.nanmean(comp_dict[var],axis=0),name=var+"_mean",dims=["pressure","latitude","longitude"],
                                            coords=[data_dict["y_wind"].pressure.values,np.arange(-7.5,8,0.5),np.arange(-7.5,8,0.5)])
        comp_dict[var+"_std"]=xr.DataArray(np.nanstd(comp_dict[var],axis=0),name=var+"_std",dims=["pressure","latitude","longitude"],
                                           coords=[data_dict["y_wind"].pressure.values,np.arange(-7.5,8,0.5),np.arange(-7.5,8,0.5)])
        del comp_dict[var]
    
    out=xr.merge([comp_dict[field] for field in list(comp_dict.keys())])
    out.attrs["number_cores"]=N
    out.to_netcdf(f"/gws/ssde/j25b/lmcs/bmaybee/aew_work/composites/DY2_CTC_{sim}_aew_{phase}_dynamics_0p5deg.nc")


def thermodynamics_composites():    
    comp_dict={}
    for var in var_list:
        comp_dict[var]=[]

    period=aew_tracks.time.values
    for t in period:
        print(t)
        
        data_dict={}
        data_dict["t_anom"]=thermo["air_temperature"].sel(time=t) - t_clim.sel(hour=pd.Timestamp(t).hour).values
        data_dict["q_anom"]=thermo["specific_humidity"].sel(time=t) - q_clim.sel(hour=pd.Timestamp(t).hour).values
        data_dict["theta_e_anom"]=thermo["equivalent_potential_temperature"].sel(time=t) - te_clim.sel(hour=pd.Timestamp(t).hour).values
        
        for s in aew_tracks.system.values:
            aew=aew_tracks.sel(time=t,system=s)
            if np.isnan(float(aew.AEW_strength)):
                continue
            if aew.AEW_lat_smooth < 7.5:
                continue
    
            for var in var_list:
                #try:
                comp_dict[var].append(composite_field(data_dict[var],aew.AEW_lat_smooth,aew.AEW_lon_smooth).values)
                #except:
                #    continue

    for var in var_list:
        comp_dict[var+"_mean"]=xr.DataArray(np.nanmean(comp_dict[var],axis=0),name=var+"_mean",dims=["pressure","latitude","longitude"],
                                            coords=[data_dict["t_anom"].pressure.values,np.arange(-7.5,8,0.5),np.arange(-7.5,8,0.5)])
        comp_dict[var+"_std"]=xr.DataArray(np.nanstd(comp_dict[var],axis=0),name=var+"_std",dims=["pressure","latitude","longitude"],
                                           coords=[data_dict["t_anom"].pressure.values,np.arange(-7.5,8,0.5),np.arange(-7.5,8,0.5)])
        del comp_dict[var]
    
    comp_dict=[comp_dict[field] for field in list(comp_dict.keys())]
    xr.merge(comp_dict).to_netcdf(f"/gws/ssde/j25b/lmcs/bmaybee/aew_work/composites/DY2_CTC_{sim}_aew_{phase}_thermodynamics_0p5deg.nc")


def mcs_composites():
    comp_dict={}
    for var in var_list:
        comp_dict[var]=[]

    period=aew_tracks.time.values
    for t in period:
        print(t)
        
        data_dict={}
        data_dict["mcs_prob"]=xr.ufuncs.sign(mcs_masks.sel(time=t,method="nearest"))
        tstampR=pd.Timestamp(t).replace(minute=30)
        try:
            date_str="2016%02d%02d"%(tstampR.month,tstampR.day)
            precip=xr.open_dataset(f"/gws/ssde/j25b/kscale/DATA/outdir_20160801T0000Z/DMn1280GAL9/channel_n2560_{sim}/precip/{date_str}_20160801T0000Z_channel_precip_hourly.nc").sel(
                                latitude=slice(0,25),longitude=slice(-40,30),time=tstampR)["precipitation_rate"]
        except:
            tstampR=tstampR-pd.Timedelta(1,"h")
            date_str="2016%02d%02d"%(tstampR.month,tstampR.day)
            precip=xr.open_dataset(f"/gws/ssde/j25b/kscale/DATA/outdir_20160801T0000Z/DMn1280GAL9/channel_n2560_{sim}/precip/{date_str}_20160801T0000Z_channel_precip_hourly.nc").sel(
                                latitude=slice(0,25),longitude=slice(-40,30),time=tstampR)["precipitation_rate"]
        data_dict["conv_prob"]=xr.ufuncs.sign(precip.where((precip>=4) & (~xr.ufuncs.isnan(data_dict["mcs_prob"]))))
        data_dict["strat_prob"]=xr.ufuncs.sign(precip.where((precip>0.5) & (precip<4) & (~xr.ufuncs.isnan(data_dict["mcs_prob"]))))
        
        for s in aew_tracks.system.values:
            aew=aew_tracks.sel(time=t,system=s)
            if np.isnan(float(aew.AEW_strength)):
                continue
            if aew.AEW_lat_smooth < 7.5:
                continue
    
            for var in var_list:
                #try:
                comp_dict[var].append(composite_field(data_dict[var],aew.AEW_lat_smooth,aew.AEW_lon_smooth).values)
                #except:
                #    continue

    for var in var_list:
        comp_dict[var]=xr.DataArray(np.nansum(comp_dict[var],axis=0)/len(period),name=var+"_mean",dims=["latitude","longitude"],
                                                        coords=[np.arange(-7.5,8,0.5),np.arange(-7.5,8,0.5)])
    
    comp_dict=[comp_dict[field] for field in list(comp_dict.keys())]
    xr.merge(comp_dict).to_netcdf(f"/gws/ssde/j25b/lmcs/bmaybee/aew_work/composites/DY2_CTC_{sim}_aew_{phase}_mcs_0p5deg.nc")
    

############################################################

    
if fields=="isetQ":    
    psize=16
    fact=int(len(np.unique(aew_tracks.time))/(psize-1))
    p=Pool(psize)
    out=p.map(isetQ_composites,np.arange(psize))
    out=xr.concat(out,dim="part")
    out.mean(dim="part").to_netcdf(f"/gws/ssde/j25b/lmcs/bmaybee/aew_work/composites/DY2_CTC_{sim}_aew_{phase}_n640_isetQ_0p2deg.nc")
    

elif fields=="mcs":
    var_list=["mcs_prob","conv_prob","strat_prob"]
    mcs_masks=xr.open_dataset(f"~/Huracan/AEW_scale_growth/MCS_tracks/Ea-waf_channel_n2560_{sim}_MCS_track_masks.nc").mcs_mask
    mcs_composites()
    

elif fields=="thermodynamics":
    var_list=["t_anom","q_anom","theta_e_anom"]
    thermo=xr.open_dataset(f"/gws/ssde/j25b/lmcs/bmaybee/aew_work/DY2_CTC_{sim}_thermo_0p5deg.nc").sel(pressure=slice(925,None))
    q_clim=thermo.specific_humidity.groupby(thermo.time.dt.hour).mean()
    t_clim=thermo.air_temperature.groupby(thermo.time.dt.hour).mean()
    te_clim=thermo.equivalent_potential_temperature.groupby(thermo.time.dt.hour).mean()

    thermodynamics_composites()

elif fields=="dynamics":
    var_list=["x_wind","y_wind","horizontal_divergence","vertical_vorticity","pv","pv_anom"]
    pv=xr.open_dataset(f"/gws/ssde/j25b/lmcs/bmaybee/aew_work/DY2_CTC_{sim}_PV_0p5deg.nc").sel(pressure=slice(925,None)).potential_vorticity
    pv_clim=pv.groupby(pv.time.dt.hour).mean()
    dyn=xr.open_dataset(f"/gws/ssde/j25b/lmcs/bmaybee/aew_work/DY2_CTC_{sim}_dynamics_0p5deg.nc").sel(pressure=slice(925,None,-1))
    dynamics_composites()