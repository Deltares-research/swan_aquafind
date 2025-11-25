import sys
import os
from datetime import datetime, timedelta
import itertools
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP
import xarray as xr

#################################
## script to setup SWAN simulations
## Requirements:
## - swan toolbox


## add swan toolbox to path
sys.path.append(
    os.path.abspath(os.path.join(r"c:\Users\ridde_mo\repos\swan-py-toolbox"))
)
sys.path.append(
    os.path.abspath(
        os.path.join(r"c:\Users\teng\00_GitHub Repositories\swan-py-toolbox")
    )
)
from swanToolBox import (
    SWANModelSetup,
    run_simulation_cluster,
    write_sp2,
    create_wind_nc,
)


###############################################################################
## user inputs
###############################################################################
## path to the main folder where the simulations will be stored
model_path = "p:\\11211806-aquafind\\WP7\\SWAN\\04_production\\01_sims\\"
## path to the template folder where the template files are stored
template_path = r"p:\11211806-aquafind\WP7\SWAN\04_production\02_scripts\template"

forcing_folder = "p:\\11211806-aquafind\\WP7\\SWAN\\04_production\\01_sims\\forcing\\"
# resolution of the model grid in meters
xresolution = 100  # horizontal resolution in x direction
yresolution = xresolution  # horizontal resolution in y direction


# output frequency in datetime.timedelta format (every hour is default in NS)
output_frequency_PNT = timedelta(minutes=10)
output_frequency_block = timedelta(hours=1)

## partition and runtime for h7
partition = "4vcpu"
runnumber = 1
runtime = timedelta(days=3, hours=12)

## number of simulations per batch script
Nbatch = 10

## simulation time
start_time_sims = datetime(2025, 1, 1, 0, 0)
end_time_sims = datetime(2025, 1, 1, 12)
spinup_time = timedelta(hours=3)
## time step
dt_minutes = 10

## discretization of the frequency and direction
flow = 0.02  ## aangepast naar lagere waarde voor Tp=20s
fhigh = 0.5
dir_res = 72

## domain
domain_length = 30  # length of the domain in km

## variations
wind_list = [10]  # [0, 5, 10, 15, 20]
wind_dir_list = [270]
offshore_wave_height_list = [1]  # [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
offshore_peak_period_list = [10]  # [5.0, 7.5, 10.0, 12.5, 15, 17.5, 20]
offshore_wave_dir_list = [270.0]
offshore_dspr_list = [20]  #!wat doen we hiermee?
bathy_list = ["constantBed"]
time_series_list = ["D"]  # ["A", "B", "C", "D", "E", "F"]

!!! quadruplets uit bij wind=0
STOPC ipv ACCUR, kijken welke defaults

# !! toevoegen initial in swan input file, dan kan runup time korter!

depth = 2000  ## diep water, diepte constantbed varieren bij complex, of gelijk naar echte bathy?


## overwrite if file already exists
overwrite = True
## constant or time varying wind conditions
wind_constant = True
## constant or time varying wave conditions
par_constant = False

###############################################################################
## make combinations
###############################################################################
comb = [
    bathy_list,
    wind_list,
    wind_dir_list,
    offshore_wave_height_list,
    offshore_peak_period_list,
    offshore_wave_dir_list,
    offshore_dspr_list,
    time_series_list,
]
combinations = list(itertools.product(*comb))

## TODO: now almost all combiantions are created but we could also add some requirements
max_steepness = 1.0


###############################################################################
## helper functions
###############################################################################
def create_batch_h7(maindir, sims, command, runnumber=0):
    ## string
    string = ""
    for ii, sim in enumerate(sims):
        string = string + "cd {} \n {}\ncd ..\n".format(sim, command)

    with open(os.path.join(maindir, "run_{}.sh".format(runnumber)), "w") as f:
        f.write(string)


###############################################################################
## create sims
###############################################################################
runnumber = 0
swan_setup_list = []
## make combinations
for ii, item in enumerate(combinations):
    ## get conditions
    bathy = item[0]
    wind = item[1]
    wind_dir = item[2]
    offshore_wave_height = item[3]
    offshore_peak_period = item[4]
    offshore_wave_dir = item[5]
    offshore_dspr = item[6]
    time_series = item[7]

    ## requirements
    steepness = offshore_wave_height / (9.81 * offshore_peak_period**2 / (2 * np.pi))
    if steepness > max_steepness:
        print(
            "Skipping steepness {:2.3f} for Hm0={:2.2f} Tp={:2.2f}".format(
                steepness, offshore_wave_height, offshore_peak_period
            )
        )
        continue



    if wind == 0 and offshore_wave_height == 0:
        print("Skipping 0m wave height with no wind")
        continue

    if offshore_wave_height == 0 and offshore_peak_period != 5 | offshore_dspr != 10:
        print(
            "Skipping 0m wave height with Tp={:2.2f} Dspr={:2.2f}".format(
                offshore_peak_period, offshore_dspr
            )
        )
        continue


#### checks die we niet doen:
    # if wind < 10 and offshore_wave_height > 2.0 and offshore_peak_period < 10:
    #     print("Skipping low wind {} for Hm0={:2.2f} and Tp={:2.2f}".format(wind, offshore_wave_height, offshore_peak_period))
    #     continue

    # if (offshore_peak_period > 10 and offshore_dspr > 10) or (
    #     offshore_peak_period <= 10 and offshore_dspr < 30
    # ):
    #     print(
    #         "Skipping dspr {} for Tp={:2.2f}".format(
    #             offshore_dspr, offshore_peak_period
    #         )
    #     )
    #     continue

    sim_name = "{}_{}_u={:2.2f}Du={:03.0f}Hm0={:2.2f}Tp={:2.2f}Dw={:03.0f}Dspr={:2.0f}".format(
        bathy,
        time_series,
        wind,
        Decimal(wind_dir).to_integral_value(ROUND_HALF_UP),
        offshore_wave_height,
        offshore_peak_period,
        Decimal(offshore_wave_dir).to_integral_value(ROUND_HALF_UP),
        offshore_dspr,
        # xresolution,
    )
    if os.path.exists(os.path.join(model_path, sim_name)) and overwrite:
        print(
            "Overwriting existing folder {}".format(os.path.join(model_path, sim_name))
        )
    elif os.path.exists(os.path.join(model_path, sim_name)) and not overwrite:
        print("Skipping folder {}".format(os.path.join(model_path, sim_name)))
        continue
    else:
        os.mkdir(os.path.join(model_path, sim_name))

    lines = [
        "Windspeed                       : {:2.2f}".format(wind),
        "Wind direction                  : {:2.2f}".format(wind_dir),
        "Wave height                     : {:2.2f}".format(offshore_wave_height),
        "Wave period                     : {:2.2f}".format(offshore_peak_period),
        "Wave direction                  : {:2.2f}".format(offshore_wave_dir),
        "Directional spreading           : {:2.2f}".format(offshore_dspr),
        "Time series                     : {}".format(time_series),
        "Bathy                           : {}".format(bathy),
    ]

    meta_string = "\n".join("!** " + line for line in lines)

    ###############################################################################
    ## define swan model setup
    ###############################################################################
    swan = SWANModelSetup(sim_name, template_path=template_path)

    swan.set_cluster(partition, runtime)

    swan.set_settings(
        {
            "mode": "NONST",
            "cdcap": 0.00275,
            "level": 0,
            "swan_module": "41.51.3_ifx2024.2.0",
            "swan_exe": "swan_4151_3_del_l64_ifx24_omp.exe",
            "coordinates": "CARTESIAN",
            "start": start_time_sims,
            "dt": dt_minutes,
            "stop": end_time_sims + spinup_time,
            "meta_string": meta_string,
        }
    )

    ###############################################################################
    ## define model domain
    ###############################################################################

    # angle
    alpc = 0
    # length of the domain in km
    xlenc = domain_length * 1000  # convert to meters
    ylenc = domain_length * 1000  # convert to meters
    ## origin
    xpc = 0
    ypc = 0
    ## grid resolution
    mxc = int(xlenc / xresolution)
    myc = int(ylenc / yresolution)

    swan.set_grid(
        "REGULAR",
        {
            "dir_res": dir_res,
            "xpc": xpc,
            "ypc": ypc,
            "alpc": alpc,
            "xlenc": xlenc,
            "ylenc": ylenc,
            "flow": flow,
            "fhigh": fhigh,
            "Nx": mxc,
            "Ny": myc,
            "grid_info": None,
        },
    )

    ###############################################################################
    ## bathymetry
    ###############################################################################
    if bathy == "constantBed":
        mxinp = 200
        myinp = 200
        bathy = np.ones((mxinp + 1, myinp + 1)) * depth
        dxinp = xlenc / (mxinp + 1)
        dyinp = ylenc / (myinp + 1)

        sim_name_bathymetry = "constant_bathy"

        np.savetxt(
            os.path.join(model_path, sim_name, f"{sim_name_bathymetry}.bot"),
            bathy,
            delimiter=" ",
            fmt="%.7e",
        )
    else:
        ## todo
        raise NotImplementedError("Only constant bathymetry is implemented so far")

    swan.set_bottom(
        "REGULAR",
        {
            "bottom_info": f"'{sim_name_bathymetry}.bot' idla=3 FREE",
            "obstacles_path": None,
            "xpinp": 0,
            "ypinp": 0,
            "alpinp": 0,
            "mxinp": mxinp,
            "myinp": myinp,
            "dxinp": dxinp,
            "dyinp": dyinp,
        },
    )

    ###############################################################################
    ## get boundary conditions
    ###############################################################################
    ## add constant values for spinup!

    date_series = [
        datetime(2025, 1, 1, 0, 0) + timedelta(minutes=i * dt_minutes)
        for i in range(
            int(
                (end_time_sims + spinup_time - start_time_sims).total_seconds()
                / 60
                / dt_minutes
            )
            + 1
        )
    ]
    if par_constant:
        Hm0_series = np.ones(len(date_series)) * offshore_wave_height  ## timeseries
        Tp_series = np.ones(len(date_series)) * offshore_peak_period
        Dir_series = np.ones(len(date_series)) * offshore_wave_dir
        Dspr_series = np.ones(len(date_series)) * offshore_dspr
    else:
        # timeseries = np.loadtxt(os.path.join('series','serie_'+time_series+ '.txt'))

        timeseries = xr.open_dataset(
            os.path.join(
                "series", "SIMPLE_HKNA_segment_MeanNorm_" + time_series + ".nc"
            )
        )
        maxnorm_timeseries = xr.open_dataset(
            os.path.join("series", "SIMPLE_HKNA_segment_MaxNorm_" + time_series + ".nc")
        )

        Hm0_series = timeseries.Hm0.values * offshore_wave_height

        Tp_series = maxnorm_timeseries.Tp.values * offshore_peak_period
        # else:
        #     Tp_series = timeseries.Tp.values * offshore_peak_period
        print("any periods above 20s?", (Tp_series > 20).any())

        ## make sure no negative values
        Hm0_series[Hm0_series < 0] = 0
        Tp_series[Tp_series < 0] = 0
        Dir_series = timeseries.Mdir.values * offshore_wave_dir
        Dspr_series = (
            np.ones(len(Hm0_series)) * offshore_dspr
        )  ## constant Dspr. ##TODO: could also be timeseries

        ## add spinup time to array
        spinup_serie = np.ones(int(spinup_time.total_seconds() / 60 / dt_minutes))
        Hm0_series = np.concatenate((spinup_serie * Hm0_series[0], Hm0_series))
        Tp_series = np.concatenate((spinup_serie * Tp_series[0], Tp_series))
        Dir_series = np.concatenate((spinup_serie * Dir_series[0], Dir_series))
        Dspr_series = np.concatenate((spinup_serie * Dspr_series[0], Dspr_series))

    plt.figure(figsize=(10, 10), layout="constrained")
    plt.subplot(4, 1, 1)
    plt.plot(date_series, Hm0_series, label="Hm0")
    plt.ylabel("$H_{m0}$")
    plt.subplot(4, 1, 2)
    plt.plot(date_series, Tp_series, label="Tp")
    plt.ylabel("$T_{p}$")
    plt.subplot(4, 1, 3)
    plt.plot(date_series, Dir_series, label="Hm0")
    plt.ylabel(r"$\theta_m$")
    plt.subplot(4, 1, 4)
    plt.plot(date_series, Dspr_series, label="Hm0")
    plt.ylabel(r"$D_{spr}$")
    plt.savefig(os.path.join(model_path, sim_name, "wave.png"))

    swan.set_bc(
        type="par",
        boundary_side="WEST",
        bc_file=os.path.join(model_path, sim_name, "wave.par"),
        dspr_setting="DEGREES",
        initial=True,
        Hm0_series=Hm0_series,
        Tp_series=Tp_series,
        Dir_series=Dir_series,
        Dspr_series=Dspr_series,
        date_series=date_series,
    )

    ##################################################################
    ## set forcings (1. current)
    ###################################################################

    if False:
        current_field_x = np.ones((mxinp + 1, myinp + 1)) * current
        current_field_y = np.ones((mxinp + 1, myinp + 1)) * 0

        current_field = current_field_x
        current_field = np.append(current_field, current_field_y, axis=0)

        sim_name_current = "constant_current"
        np.savetxt(
            os.path.join(model_path, sim_name, f"{sim_name_current}.cur"),
            current_field,
            delimiter=" ",
            fmt="%.7e",
        )

        swan.set_flow(f"{sim_name_current}.cur", currents_nc=False)

    ##################################################################
    ## set forcings (2. wind)
    ##################################################################

    if wind_constant:
        swan.set_wind(type="constant", wind_speed=wind, wind_dir=wind_dir)

    else:
        assert (
            par_constant == False
        ), "If time varying wind is used, par_constant must be False for wave conditions"
        xgrid, ygrid = np.meshgrid(
            np.linspace(0, xlenc, mxc + 1), np.linspace(0, ylenc, myc + 1)
        )

        uwind = np.zeros((len(date_series), ygrid.shape[0], xgrid.shape[1]))
        vwind = np.zeros((len(date_series), ygrid.shape[0], xgrid.shape[1]))

        wind_speed_series = timeseries[:, 3]  # * wind
        wind_dir_serie = (
            np.ones_like(wind_speed_series) * wind_dir
        )  ##TODO: constant wind direction in time
        ## add spinup time to array
        wind_speed_series = np.concatenate(
            (spinup_serie * wind_speed_series[0], wind_speed_series)
        )
        wind_dir_serie = np.concatenate(
            (spinup_serie * wind_dir_serie[0], wind_dir_serie)
        )
        for jj, date in enumerate(date_series):

            uwind[jj, :, :] = wind_speed_series[jj] * np.cos(
                np.deg2rad(wind_dir_serie[jj])
            )
            vwind[jj, :, :] = wind_speed_series[jj] * np.sin(
                np.deg2rad(wind_dir_serie[jj])
            )

        create_wind_nc(
            forcing_folder,
            time_series,
            xgrid,
            ygrid,
            uwind,
            vwind,
            date_series,
            fig=False,
        )

        swan.set_wind(
            type="field",
            wind_path="..\\forcing\\" + time_series + "_wind.nc",
            start=start_time_sims,
            dt=60,
            stop=end_time_sims + spinup_time,
            wind_factor=wind,
        )
    ##################################################################
    ## set forcings (3. Water level)
    ##################################################################

    if False:
        swan.set_water_level(
            water_level_path=os.path.join(model_path, sim_name, "water_level.nc"),
            start=start_time_sims,
            dt=dt_minutes,
            stop=end_time_sims,
        )

    #######################################################################
    ## set physics
    ########################################################################

    swan.set_physics(
        {
            "GEN3": "ST6 5.6E-6 17.5E-5 VECTAU U10P 31. AGROW",
            "SSWELL": "",
            "QUAD": "iquad=3",
            "FRIC": "JONSWAP 0.038",
            "TRIAD": "ITRIAD=11 TRFAC=0.1 CUTFR=2.5",
            "BREA": "CONST   1.0    0.73",
        }
    )

    #######################################################################
    ## set output
    #######################################################################

    output_x = domain_length * 1000
    output_y = domain_length * 1000 / 2  # Place input buoy at the center of the domain

    # create line with output points in centre of domain
    Noutput = 30
    dx_output = domain_length * 1000 / Noutput
    coordinates = []
    for i in range(Noutput):
        x_loc = i * dx_output
        coordinates.extend(
            [
                (
                    x_loc,
                    output_y,
                    "line of output points at y={:2.1f} m".format(output_y),
                ),
            ]
        )

    # create second line with output points north of the centre line
    Noutput_NS = 10
    dx_output_NS = domain_length * 1000 / Noutput_NS
    output_y_N = output_y + output_y / 2

    for i in range(Noutput_NS):
        x_loc = i * dx_output_NS
        coordinates.extend(
            [
                (
                    x_loc,
                    output_y_N,
                    "line of output points at y={:2.1f} m".format(output_y_N),
                ),
            ]
        )
    # create second line with output points south of the centre line
    output_y_S = output_y - output_y / 2
    for i in range(Noutput_NS):
        x_loc = i * dx_output_NS
        coordinates.extend(
            [
                (
                    x_loc,
                    output_y_S,
                    "line of output points at y={:2.1f} m".format(output_y_S),
                ),
            ]
        )

    # Write to a text file
    output_location = "points.PNT"
    with open(
        os.path.join(model_path, sim_name, output_location),
        "w",
    ) as file:
        for x, y, comment in coordinates:
            file.write(f"{x}   {y}\t# {comment}\n")

    swan.set_output_block(
        variables=[
            "XP",
            "YP",
            "HSIG",
            "BOTLEV",
            "HSWELL",
            "TMM10",
            "TPS",
            "DIR",
            "DSPR",
            "WATLEV",
            "VEL",
            "WIND",
        ],
        dt=output_frequency_block,
    )

    swan.set_output_pnt(
        "P2",
        variables=[
            "TIME",
            "XP",
            "YP",
            "DEP",
            "HSIG",
            "HSWELL",
            "TMM10",
            "TM02",
            "TPS",
            "DIR",
            "FDIR",
            "DSPR",
            "WIND",
            "WATLEV",
            "VEL",
            "TM01",
        ],
        dt=output_frequency_PNT,
        file=f"{output_location}",
        spec1D=False,
        spec2D=True,
    )

    #######################################################################
    ## write model
    #######################################################################
    swan.create_model(model_path)

    swan_setup_list.append(os.path.join(sim_name))
    #######################################################################
    ## make batch script
    #######################################################################
    if len(swan_setup_list) > Nbatch:
        create_batch_h7(
            os.path.join(model_path),
            swan_setup_list,
            "dos2unix swanrunH7_docker.sh & sbatch swanrunH7_docker.sh",
            runnumber=runnumber,
        )
        ## clear list
        swan_setup_list = []
        ## update runnumber
        runnumber += 1

if len(swan_setup_list) > 0:
    create_batch_h7(
        os.path.join(model_path),
        swan_setup_list,
        "dos2unix swanrunH7_docker.sh & sbatch swanrunH7_docker.sh",
        runnumber=runnumber,
    )
