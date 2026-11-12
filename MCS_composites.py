from utils import * 



# these are L/2
scales=[125000.,  250000., 500000., 1000000.]

#########################################################################################
parser=argparse.ArgumentParser()
parser.add_argument("-s", "--sim", required=False, default='RAL3p2')
parser.add_argument("-v","--var", required=False, default="ushear650_850")
parser.add_argument("-f","--field", required=False, default="isetQ")
parser.add_argument("-scale","--scale",required=False, default=250, type=int) # this takes L
args = parser.parse_args()
#########################################################################################

sim=args.sim
tercile_var=args.var
scale=args.scale/2 * 1000
field=args.field

if sim=="RAL3p2":
    sim2="RAL3"
elif sim=="GAL9":
    sim2="GAL9"

WAf_trop5_MCS=pd.read_csv(f"Tables/Channel_n2560_{sim}_Wafrica_MCS_table_jmn_empbt.csv")
WAf_trop5_MCS=WAf_trop5_MCS[WAf_trop5_MCS.rain_max>1]

Qprofs=xr.open_dataset(f"/gws/ssde/j25b/lmcs/bmaybee/aew_work/Channel_{sim}_ea-waf_Q_diagnostics_allZ_plevs.nc").rename(
                 {"level":"pressure"}).assign_coords(pressure=[100., 150., 200., 250., 300., 400., 500., 600., 700., 850., 925.])
period=pd.date_range(start="2016-08-01",end="2016-09-09")


def dynamics_composites():    
    comp_dict={}
    for var in var_list:
        comp_dict[var]=[]
    MCS_data=WAf_trop5_MCS[(WAf_trop5_MCS["hour"].isin([15,18,21])) & (WAf_trop5_MCS["tminlat"]<20)]
    pvals=dyn["x_wind"].pressure.values

    period=pd.to_datetime(MCS_data.date.values,format="%Y%m%d_%H:%M").unique() - pd.Timedelta(2,"m")
    N=0
    for j, t in enumerate(period):
        print(t)
        day_data=MCS_data[(MCS_data["day"]==t.day) & (MCS_data["month"]==t.month) & (MCS_data["hour"]==t.hour)]
        
        data_dict={}
        data_dict["x_wind"]=dyn["x_wind"].sel(time=t)
        data_dict["y_wind"]=dyn["y_wind"].sel(time=t)
        data_dict["horizontal_divergence"]=dyn["horizontal_divergence"].sel(time=t)
        data_dict["vertical_vorticity"]=dyn["vertical_vorticity"].sel(time=t)
        data_dict["pv"]=pv.sel(time=t)
        data_dict["pv_anom"]=data_dict["pv"] - pv_clim.sel(hour=pd.Timestamp(t).hour).values
        
        for i in range(len(day_data)):
            N+=1
            for var in var_list:
                comp_dict[var].append(composite_field(data_dict[var],day_data["tminlat"].iloc[i],day_data["tminlon"].iloc[i],ll_border=4,size=17).values)
                

    N=len(comp_dict[var])
    for var in var_list:
        comp_dict[var+"_mean"]=xr.DataArray(np.nanmean(comp_dict[var],axis=0),name=var+"_mean",dims=["pressure","latitude","longitude"],
                                            coords=[pvals,np.arange(-4,4.5,0.5),np.arange(-4,4.5,0.5)])
        comp_dict[var+"_std"]=xr.DataArray(np.nanstd(comp_dict[var],axis=0),name=var+"_std",dims=["pressure","latitude","longitude"],
                                            coords=[pvals,np.arange(-4,4.5,0.5),np.arange(-4,4.5,0.5)])
        del comp_dict[var]
    
    out=xr.merge([comp_dict[field] for field in list(comp_dict.keys())])
    out.attrs["number_cores"]=N
    out.to_netcdf(f"/gws/ssde/j25b/lmcs/bmaybee/aew_work/composites/DY2_CTC_{sim}_MCS_dynamics_0p5deg.nc")



def deltaIset_composites(MCS_data):
    MCS_data=MCS_data[(MCS_data["hour"].isin([18,21])) & (MCS_data["tminlat"]<20)]
    comp={}
    for ter in ["deltaDl_250-500","deltaDl_1000-2000"]:
        comp[ter]=np.zeros((11,41,41))
        comp[ter+"_count"]=np.zeros((11,41,41))
        
    for j, date in enumerate(period):
        print(date)
        Dl=[]
        for ffile in glob.glob(f"/gws/ssde/j25b/kscale/LoSSETT_data/DYAMOND_Summer/global_n1280_GAL9/channel_n2560_{sim}/n640/channel_n2560_{sim
                                }.n1280_GAL9_inter_scale_transfer_of_kinetic_energy_Lmin_00032_Lmax_02500_2016{date.month:02d}{date.day:02d}T12_p*_n1280_embedded_n640.nc"):
                ds=xr.open_dataset(ffile).sel(longitude=slice(-20,30),latitude=slice(0,25),length_scale=[125000,250000,500000,1000000]).Dl_u
                Dl.append(ds.interp(latitude=np.arange(-5,25.1,0.2),longitude=np.arange(-40,35.1,0.2)))           
        Dl=xr.concat(Dl,dim="pressure").transpose("time","pressure","length_scale","latitude","longitude")

        deltaDl_250_500 = Dl.sel(length_scale=250000) - Dl.sel(length_scale=125000)
        deltaDl_1000_2000 = Dl.sel(length_scale=1000000) - Dl.sel(length_scale=500000)
        day_data=MCS_data[(MCS_data["day"]==date.day) & (MCS_data["month"]==date.month)]
        for i in range(len(day_data)):
            hr=day_data["hour"].iloc[i]
            arr=composite_field(deltaDl_250_500.sel(time=date.replace(hour=hr)),day_data["tminlat"].iloc[i],day_data["tminlon"].iloc[i],ll_border=4,size=41)
            comp["deltaDl_250-500"] = np.nansum((comp["deltaDl_250-500"],arr), axis=0)
            comp["deltaDl_250-500_count"] = np.nansum((comp["deltaDl_250-500_count"], (~np.isnan(arr)).astype(int)),axis=0)
            
            arr=composite_field(deltaDl_1000_2000.sel(time=date.replace(hour=hr)),day_data["tminlat"].iloc[i],day_data["tminlon"].iloc[i],ll_border=4,size=41)
            comp["deltaDl_1000-2000"] = np.nansum((comp["deltaDl_1000-2000"],arr), axis=0)
            comp["deltaDl_1000-2000_count"] = np.nansum((comp["deltaDl_1000-2000_count"], (~np.isnan(arr)).astype(int)),axis=0)
            
    out=[]
    for ter in ["deltaDl_250-500","deltaDl_1000-2000"]:
        ds=xr.DataArray(comp[ter]/comp[ter+"_count"],coords=[Dl.pressure.values,np.arange(-4,4.1,0.2),np.arange(-4,4.1,0.2)],
                                                        dims=["pressure","lat","lon"],name=ter)
        out.append(ds)
        
    return xr.merge(out)


def vertical_Dl_composites(MCS_data,scale,shear_tercile=None,shear_var="ushear650_850"):
    MCS_data=MCS_data[(MCS_data["hour"].isin([15,18,21])) & (MCS_data["tminlat"]<20)]
    env_dict={}
    env_dict["all"]=MCS_data
    env_dict["t1"]=MCS_data[MCS_data[shear_var] < MCS_data[shear_var].quantile(0.33)]
    env_dict["t2"]=MCS_data[(MCS_data[shear_var] >= MCS_data[shear_var].quantile(0.33)) & (MCS_data[shear_var] <= MCS_data[shear_var].quantile(0.66))]
    env_dict["t3"]=MCS_data[MCS_data[shear_var] > MCS_data[shear_var].quantile(0.66)]

    
    terciles=["all","t1","t2","t3"]
    # plevs; axis1 length: 15 for profile + surface, 14 for full profile, 8 for 925 - 500
    comp={}
    for ter in terciles:
        comp["Dl_"+ter]=np.zeros((11,41,41))
        comp["Dl_"+ter+"_count"]=np.zeros((11,41,41))
        comp["Qtot_"+ter]=np.zeros((11,17,17))
        comp["Qtot_"+ter+"_count"]=np.zeros((11,17,17))

    for j, date in enumerate(period):
        print(date)
        
        Dt=[]
        for ffile in glob.glob(f"/gws/ssde/j25b/kscale/LoSSETT_data/DYAMOND_Summer/global_n1280_GAL9/channel_n2560_{sim}/n640/channel_n2560_{sim
                                }.n1280_GAL9_inter_scale_transfer_of_kinetic_energy_Lmin_00032_Lmax_02500_2016{date.month:02d}{date.day:02d}T12_p*_n1280_embedded_n640.nc"):
                ds=xr.open_dataset(ffile).sel(longitude=slice(-20,30),latitude=slice(0,25),length_scale=scale).Dl_u
                Dt.append(ds.interp(latitude=np.arange(-5,25.1,0.2),longitude=np.arange(-40,35.1,0.2)))           
        Dt=xr.concat(Dt,dim="pressure").transpose("time","pressure","latitude","longitude")
        Qt=Qprofs.Q_tot.sel(time=slice(date.replace(hour=15),date.replace(hour=21)),pressure=slice(None,None,-1)).transpose("time","pressure","latitude","longitude")

        #Qt=Q.Q_tot.sel(time=date.replace(hour=hr))
        for ter in terciles:
            MCS_data=env_dict[ter]
            day_data=MCS_data[(MCS_data["day"]==date.day) & (MCS_data["month"]==date.month)]
            for i in range(len(day_data)):
                hr=day_data["hour"].iloc[i]
                arr=composite_field(Dt.sel(time=date.replace(hour=hr)),day_data["tminlat"].iloc[i],day_data["tminlon"].iloc[i],ll_border=4,size=41)
                comp["Dl_"+ter] = np.nansum((comp["Dl_"+ter],arr), axis=0)
                comp["Dl_"+ter+"_count"] = np.nansum((comp["Dl_"+ter+"_count"], (~np.isnan(arr)).astype(int)),axis=0)
                
                arr=composite_field(Qt.sel(time=date.replace(hour=hr)),day_data["tminlat"].iloc[i],day_data["tminlon"].iloc[i],ll_border=4,size=17)
                comp["Qtot_"+ter] = np.nansum((comp["Qtot_"+ter],arr), axis=0)
                comp["Qtot_"+ter+"_count"] = np.nansum((comp["Qtot_"+ter+"_count"], (~np.isnan(arr)).astype(int)),axis=0)
                
    out=[]
    for ter in terciles:
        ds=xr.DataArray(comp["Dl_"+ter]/comp["Dl_"+ter+"_count"],coords=[Dt.pressure.values,np.arange(-4,4.1,0.2),np.arange(-4,4.1,0.2)],
                                                        dims=["pressure","lat","lon"],name="Dl_"+ter)
        dsQ=xr.DataArray(comp["Qtot_"+ter]/comp["Qtot_"+ter+"_count"],coords=[Dt.pressure.values,np.arange(-4,4.1,0.5),np.arange(-4,4.1,0.5)],
                                                        dims=["pressure","latitude","longitude"],name="Qtot_"+ter)
        out.append(ds)
        out.append(dsQ)
        
    return xr.merge(out)


if field=="isetQ":
    out=vertical_Dl_composites(WAf_trop5_MCS,scale=scale,shear_var=tercile_var)
    if tercile_var=="ushear650_850":
        tercile_var="shear"
    out.to_netcdf(f"/gws/ssde/j25b/lmcs/bmaybee/aew_work/composites/DY2_CTC_{sim}_MCS_{tercile_var}Ter_n640_L={args.scale}_isetQ_0p2deg.nc")

elif field=="deltaIset":
    out=deltaIset_composites(WAf_trop5_MCS)
    out.to_netcdf(f"/gws/ssde/j25b/lmcs/bmaybee/aew_work/composites/DY2_CTC_{sim}_MCS_deltaIset_0p2deg.nc")

elif field=="dynamics":
    var_list=["x_wind","y_wind","horizontal_divergence","vertical_vorticity","pv","pv_anom"]
    pv=xr.open_dataset(f"/gws/ssde/j25b/lmcs/bmaybee/aew_work/DY2_CTC_{sim}_PV_0p5deg.nc").sel(pressure=slice(925,None)).potential_vorticity
    pv_clim=pv.groupby(pv.time.dt.hour).mean()
    dyn=xr.open_dataset(f"/gws/ssde/j25b/lmcs/bmaybee/aew_work/DY2_CTC_{sim}_dynamics_0p5deg.nc").sel(pressure=slice(925,None,-1))
    dynamics_composites()