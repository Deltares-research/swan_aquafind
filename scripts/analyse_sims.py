import sys
import os
from datetime import datetime, timedelta
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm
from decimal import Decimal, ROUND_HALF_UP
import xarray as xr
import math

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
from swanToolBox import extract_accuracy_data

from swanToolBox.swan_postprocessing.read_SWAN import (
    read_SWAN_tab,
)

###############################################################################
## user inputs
###############################################################################
## path to the main folder where the simulations will be stored
model_path = "p:\\11211806-aquafind\\WP7\\SWAN\\04_production\\01_sims\\"
fig_path = "p:\\11211806-aquafind\\WP7\\SWAN\\04_production\\00_results\\"

## variations
wind_list = [0, 5, 10, 15, 20]
wind_dir_list = [270]
offshore_wave_height_list = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
offshore_peak_period_list = [5.0, 7.5, 10.0, 12.5, 15, 17.5, 20]
offshore_wave_dir_list = [270.0]
offshore_dspr_list = [10, 30]
bathy_list = ["constantBed"]
time_series_list = ["A", "B", "C", "D", "E"]

wind_list = [0, 10, 20]  # [0, 5, 10, 15, 20]
wind_dir_list = [270]
offshore_wave_height_list = [0, 0.5, 3.0]  # [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
offshore_peak_period_list = [5, 20]  # [5.0, 7.5, 10.0, 12.5, 15, 17.5, 20]
offshore_wave_dir_list = [270.0]
offshore_dspr_list = [10, 30]  #!wat doen we hiermee?
bathy_list = ["constantBed"]
time_series_list = ["B", "C"]  # ["A", "B", "C", "D", "E", "F"]

analyse_map = True
analyse_points = True
spec_output = True
write_netCDF = True

# input_x = 0  ! of verderop in het domein? nog niet in gebruik!
output_y = 15000

depth = 2000

spin_up_time = np.timedelta64(3, "h")

"""
TODO
- Kijken naar wat er gebeurt met Dspr, zakt enorm in. Check voor maken?
- DONE: output_xy! afhankelijk maken van de Tp, dus per combinatie bepalen o.b.v. groepsnelheid*30min
- ..
- DONE: Domein verlengen, zie create_sims.
- 

"""

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


###############################################################################
## helper functions
###############################################################################
def groupspeed(h, T):
    g = 9.81
    deepL = g * T**2 / (2 * np.pi)
    for ii in range(1000):
        if ii == 0:
            newL = deepL * np.tanh(2 * np.pi * h / deepL)
        rest = newL - deepL * np.tanh(2 * np.pi * h / newL)
        # if rest < 0.1:
        #     break
        newL = deepL * np.tanh(2 * np.pi * h / newL)
    k = 2 * np.pi / newL
    n = 0.5 * (1 + 2 * k * h / np.sinh(2 * k * h))
    c = newL / T
    cg = n * c
    return cg


###############################################################################
## analyse sims
###############################################################################

failed_sims = []
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

    sim_name = (
        "{}_{}_u={:2.2f}Du={:03.0f}Hm0={:2.2f}Tp={:2.2f}Dw={:03.0f}Dspr={:2.0f}".format(
            bathy,
            time_series,
            wind,
            Decimal(wind_dir).to_integral_value(ROUND_HALF_UP),
            offshore_wave_height,
            offshore_peak_period,
            Decimal(offshore_wave_dir).to_integral_value(ROUND_HALF_UP),
            offshore_dspr,
        )
    )

    if not os.path.exists(os.path.join(model_path, sim_name, "output")):
        print("cannot find: {}".format(os.path.join(model_path, sim_name, "output")))
        failed_sims.append((sim_name, "No output folder"))
        continue

    path_output = os.path.join(model_path, sim_name, "output")
    ## create fig path
    if not os.path.exists(os.path.join(fig_path, sim_name)):
        os.mkdir(os.path.join(fig_path, sim_name))
    # try:
    ###############################################################################
    ## spectrum output
    ###############################################################################
    if spec_output:
        swan_spec = xr.open_dataset(
            os.path.join(path_output, "SPEC_p2.nc"), engine="netcdf4"
        )

        fig, ax = plt.subplots(2, 3, figsize=[15, 8], layout="constrained")
        time_index0 = 10
        time_index1 = -20
        time_index2 = -10

        loc_index1 = 0
        loc_index2 = -1

        dir_array = swan_spec.direction.values  # np.rad2deg(swan_spec.direction.values)
        dir_array[np.where(dir_array < offshore_wave_dir - 180)] += 360

        sorted_indices = np.argsort(dir_array)
        dir_array_sorted = dir_array[sorted_indices]

        bounds = [0, 0.05, 0.1, 0.5, 1, 5, 10, 25, 100]
        cmap = plt.get_cmap("jet")
        norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)

        plt.subplot(2, 3, 1)
        loc_index = 0
        plt.contourf(
            swan_spec.frequency.values,
            dir_array_sorted,
            swan_spec.density.values[time_index0, loc_index1, ::].T[sorted_indices, :],
            levels=bounds,
            cmap=cmap,
            norm=norm,
        )
        plt.xlabel("$f$ [$Hz$]")
        plt.ylabel(r"$\theta$ [$^\circ$]")
        cbar = plt.colorbar(ticks=bounds)
        cbar.ax.set_yticklabels([str(b) for b in bounds])
        plt.xlim(0, 0.5)
        plt.title(
            "$x={:2.2f}m$ {}".format(
                swan_spec.x[loc_index1].values,
                np.datetime_as_string(swan_spec.time[time_index0].values, unit="h"),
            )
        )

        plt.subplot(2, 3, 4)

        plt.contourf(
            swan_spec.frequency.values,
            dir_array_sorted,
            swan_spec.density.values[time_index0, loc_index2, ::].T[sorted_indices, :],
            levels=bounds,
            cmap=cmap,
            norm=norm,
        )
        plt.xlabel("$f$ [$Hz$]")
        plt.ylabel(r"$\theta$ [$^\circ$]")
        cbar = plt.colorbar(ticks=bounds)
        cbar.ax.set_yticklabels([str(b) for b in bounds])
        plt.xlim(0, 0.5)
        plt.title(
            "$x={:2.2f}m$ {}".format(
                swan_spec.x[loc_index2].values,
                np.datetime_as_string(swan_spec.time[time_index0].values, unit="h"),
            )
        )

        plt.subplot(2, 3, 2)
        loc_index = 0
        plt.contourf(
            swan_spec.frequency.values,
            dir_array_sorted,
            swan_spec.density.values[time_index1, loc_index1, ::].T[sorted_indices, :],
            levels=bounds,
            cmap=cmap,
            norm=norm,
        )
        plt.xlabel("$f$ [$Hz$]")
        plt.ylabel(r"$\theta$ [$^\circ$]")
        cbar = plt.colorbar(ticks=bounds)
        cbar.ax.set_yticklabels([str(b) for b in bounds])
        plt.xlim(0, 0.5)
        plt.title(
            "$x={:2.2f}m$ {}".format(
                swan_spec.x[loc_index1].values,
                np.datetime_as_string(swan_spec.time[time_index1].values, unit="h"),
            )
        )

        plt.subplot(2, 3, 5)

        plt.contourf(
            swan_spec.frequency.values,
            dir_array_sorted,
            swan_spec.density.values[time_index1, loc_index2, ::].T[sorted_indices, :],
            levels=bounds,
            cmap=cmap,
            norm=norm,
        )
        plt.xlabel("$f$ [$Hz$]")
        plt.ylabel(r"$\theta$ [$^\circ$]")
        cbar = plt.colorbar(ticks=bounds)
        cbar.ax.set_yticklabels([str(b) for b in bounds])
        plt.xlim(0, 0.5)
        plt.title(
            "$x={:2.2f}m$ {}".format(
                swan_spec.x[loc_index2].values,
                np.datetime_as_string(swan_spec.time[time_index1].values, unit="h"),
            )
        )

        plt.subplot(2, 3, 3)

        plt.contourf(
            swan_spec.frequency.values,
            dir_array_sorted,
            swan_spec.density.values[time_index2, loc_index1, ::].T[sorted_indices, :],
            levels=bounds,
            cmap=cmap,
            norm=norm,
        )
        plt.xlabel("$f$ [$Hz$]")
        plt.ylabel(r"$\theta$ [$^\circ$]")
        cbar = plt.colorbar(ticks=bounds)
        cbar.ax.set_yticklabels([str(b) for b in bounds])
        plt.xlim(0, 0.5)
        plt.title(
            "$x={:2.2f}m$ {}".format(
                swan_spec.x[loc_index1].values,
                np.datetime_as_string(swan_spec.time[time_index2].values, unit="h"),
            )
        )

        plt.subplot(2, 3, 6)

        plt.contourf(
            swan_spec.frequency.values,
            dir_array_sorted,
            swan_spec.density.values[time_index2, loc_index2, ::].T[sorted_indices, :],
            levels=bounds,
            cmap=cmap,
            norm=norm,
        )
        plt.xlabel("$f$ [$Hz$]")
        plt.ylabel(r"$\theta$ [$^\circ$]")
        cbar = plt.colorbar(ticks=bounds)
        cbar.ax.set_yticklabels([str(b) for b in bounds])
        plt.xlim(0, 0.5)
        plt.title(
            "$x={:2.2f}m$ {}".format(
                swan_spec.x[loc_index2].values,
                np.datetime_as_string(swan_spec.time[time_index2].values, unit="h"),
            )
        )
        plt.savefig(os.path.join(fig_path, sim_name, "spec.png"))
        plt.close(fig)
    ##################################################################################
    ## map output
    ##################################################################################
    if analyse_map:
        map = xr.open_dataset(os.path.join(path_output, "swan2D.nc"), engine="netcdf4")
        tab_dummy = pd.read_csv(
            os.path.join(path_output, "POINTS_P2.TAB"),
            header=None,
            skiprows=7,
            delimiter=r"\s+",
        )

        ###############################################################
        fig, ax = plt.subplots(figsize=[12, 12])
        plt.pcolormesh(map.x.values, map.y.values, -map.botl.values[1, :, :])
        cbar = plt.colorbar()
        cbar.set_label("botl [m+SWL]")
        plt.plot(
            tab_dummy.iloc[:, 1], tab_dummy.iloc[:, 2], "k.", label="buoy locations"
        )
        plt.axis("equal")
        plt.xlabel("RD x (m)")
        plt.ylabel("RD y (m)")
        plt.legend()
        plt.savefig(os.path.join(fig_path, sim_name, "bed.png"))
        plt.close(fig)
        #############
        maptime = map.time.values
        maptime_first = maptime[3]
        fig, ax = plt.subplots(figsize=[12, 12])
        plt.pcolor(map.x.values, map.y.values, map.hs.values[3, :, :])
        step = 10
        plt.quiver(
            map.x.values[::step],
            map.y.values[::step],
            -1 * np.cos(np.deg2rad(map.theta0.values[3, ::step, ::step] - 90)),
            -1 * np.sin(np.deg2rad(map.theta0.values[3, ::step, ::step] - 90)),
        )
        cbar = plt.colorbar()
        cbar.set_label("Hs [m]")
        plt.plot(
            tab_dummy.iloc[:, 1], tab_dummy.iloc[:, 2], "k.", label="buoy locations"
        )
        plt.xlabel("RD x (m)")
        plt.ylabel("RD y (m)")
        plt.title("Hs" + str(pd.to_datetime(maptime_first)))
        plt.legend()
        plt.axis("equal")
        plt.savefig(os.path.join(fig_path, sim_name, "Hs_t0_map.png"))
        plt.close(fig)
        maptime_last = maptime[-1]
        fig, ax = plt.subplots(figsize=[12, 12])
        plt.pcolor(map.x.values, map.y.values, map.hs.values[-1, :, :])
        step = 10
        plt.quiver(
            map.x.values[::step],
            map.y.values[::step],
            -1 * np.cos(np.deg2rad(map.theta0.values[-1, ::step, ::step] - 90)),
            -1 * np.sin(np.deg2rad(map.theta0.values[-1, ::step, ::step] - 90)),
        )
        cbar = plt.colorbar()
        cbar.set_label("Hs [m]")
        plt.plot(
            tab_dummy.iloc[:, 1], tab_dummy.iloc[:, 2], "k.", label="buoy locations"
        )
        plt.xlabel("RD x (m)")
        plt.ylabel("RD y (m)")
        plt.title("Hs" + str(pd.to_datetime(maptime_last)))
        plt.legend()
        plt.axis("equal")
        plt.savefig(os.path.join(fig_path, sim_name, "Hs_map.png"))
        plt.close(fig)
        #############
        maptime = map.time.values
        maptime_last = maptime[-1]
        fig, ax = plt.subplots(figsize=[12, 12])
        plt.pcolor(map.x.values, map.y.values, map.spread.values[-1, :, :])
        cbar = plt.colorbar()
        cbar.set_label(r"Dspr [$^\circ$]")
        # plt.plot(tab_dummy.iloc[:, 1], tab_dummy.iloc[:, 2], "k.", label="buoy locations")
        plt.xlabel("RD x (m)")
        plt.ylabel("RD y (m)")
        plt.title("Dspr " + str(pd.to_datetime(maptime_last)))
        plt.legend()
        plt.axis("equal")
        plt.savefig(os.path.join(fig_path, sim_name, "Dspr_map.png"))
        plt.close(fig)
        #############

    ##################################################################################
    ## convergence
    ##################################################################################
    time_steps, accur_max, accur_mean, accuracy_values = extract_accuracy_data(
        os.path.join(path_output, "..", "PRINT")
    )
    iterations = np.zeros(len(time_steps))
    for jj, item in enumerate(accuracy_values):
        iterations[jj] = len(accuracy_values[jj])
    # Plotting the accuracy over time
    fig = plt.figure(figsize=(12, 6), layout="constrained")
    plt.subplot(2, 1, 1)
    plt.plot(
        time_steps,
        accur_max,
        marker="o",
        linestyle="-",
        color="blue",
        label="Max Accuracy",
    )
    plt.plot(
        time_steps,
        accur_mean,
        marker="o",
        linestyle="-",
        color="red",
        label="Mean Accuracy",
    )
    plt.title("Accuracy Over Time")
    plt.ylabel("Accuracy (%)")
    plt.xticks([])
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, iterations, marker="o")
    plt.ylabel("Iterations")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(fig_path, sim_name, "accur.png"))
    plt.close(fig)

    output_x = math.floor(groupspeed(depth, offshore_peak_period) * 1800 / 1000) * 1000
    # output_x = 22500
    ##################################################################################
    ## points output
    ##################################################################################
    if analyse_points:
        tab_data = read_SWAN_tab(
            os.path.join(path_output, "POINTS_P2.TAB")
        )  # does not work perfectly

        forced_conditions = np.loadtxt(
            os.path.join(path_output, "..", "wave.par"), skiprows=1
        )

        fig = plt.figure(figsize=[12, 12], layout="constrained")

        ax = plt.subplot(2, 2, 1)
        ax.plot(
            tab_data.Time.values,
            tab_data.Hsig.values[
                np.where(tab_data.Xp == output_x)[0][0],
                np.where(tab_data.Yp == output_y)[0][0],
                :,
            ],
            label="model x = {:2.2f}".format(output_x),
        )
        ax.plot(
            tab_data.Time.values,
            tab_data.Hsig.values[
                np.where(tab_data.Xp == 0)[0][0],
                np.where(tab_data.Yp == output_y)[0][0],
                :,
            ],
            label="model x = 0",
        )
        ax.plot(
            tab_data.Time.values,
            forced_conditions[:, 1],
            "k--",
            label="offshore Hm0",
        )
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_ylabel("$H_{m0}$ [$m$]")
        ax.legend()

        ax = plt.subplot(2, 2, 2)
        ax.plot(
            tab_data.Time.values,
            tab_data.TPsmoo.values[
                np.where(tab_data.Xp == output_x)[0][0],
                np.where(tab_data.Yp == output_y)[0][0],
                :,
            ],
        )
        ax.plot(
            tab_data.Time.values,
            tab_data.TPsmoo.values[
                np.where(tab_data.Xp == 0)[0][0],
                np.where(tab_data.Yp == output_y)[0][0],
                :,
            ],
            label="model x = 0",
        )
        ax.plot(tab_data.Time.values, forced_conditions[:, 2], "k--")
        ax.set_ylim(0)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_ylabel("$T_{p}$ [$s$]")

        ax = plt.subplot(2, 2, 3)
        ax.plot(
            tab_data.Time.values,
            tab_data.Dir.values[
                np.where(tab_data.Xp == output_x)[0][0],
                np.where(tab_data.Yp == output_y)[0][0],
                :,
            ],
        )

        ax.plot(
            tab_data.Time.values,
            tab_data.Dir.values[
                np.where(tab_data.Xp == 0)[0][0],
                np.where(tab_data.Yp == output_y)[0][0],
                :,
            ],
            label="model x = 0",
        )
        ax.plot(tab_data.Time.values, forced_conditions[:, 3], "k--")
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylabel(r"$\theta_m$ [$^\circ$]")

        ax = plt.subplot(2, 2, 4)
        ax.plot(
            tab_data.Time.values,
            tab_data.Dspr.values[
                np.where(tab_data.Xp == output_x)[0][0],
                np.where(tab_data.Yp == output_y)[0][0],
                :,
            ],
        )
        ax.plot(
            tab_data.Time.values,
            tab_data.Dspr.values[
                np.where(tab_data.Xp == 0)[0][0],
                np.where(tab_data.Yp == output_y)[0][0],
                :,
            ],
            label="model x = 0",
        )
        # ax.plot(
        #     tab_data.Time.values,
        #     tab_data.Dspr.values[
        #         np.where(tab_data.Xp == 1000)[0][0],
        #         np.where(tab_data.Yp == output_y)[0][0],
        #         :,
        #     ],
        #     label="model x = 1500",
        # )
        ax.plot(tab_data.Time.values, forced_conditions[:, 4], "k--")
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_ylabel(r"$D_{spr}$ [$^\circ$]")
        fig.savefig(os.path.join(fig_path, sim_name, "point_output.png"))
        plt.close(fig)

    #################################################
    ## netcdf
    #################################################
    if spec_output and write_netCDF:

        ## remove spinup time
        id_wanted = (swan_spec.time.values - swan_spec.time[0].values) > spin_up_time
        ## select only points with the same y location
        y_index = np.where(
            (swan_spec.y.values == output_y)
            # & ((swan_spec.x.values == output_x) | (swan_spec.x.values == 0))
        )[0]

        E = swan_spec.density.isel(points=y_index).sel(time=id_wanted).values

        Tp_at_outputx = tab_data.TPsmoo.values[
            :,
            np.where(tab_data.Yp == output_y)[0][0],
            :,
        ]
        distance_array = np.floor(groupspeed(depth, Tp_at_outputx) * 1800 / 1000) * 1000

        swan_spec["xwnd"] = swan_spec["xwnd"].fillna(0)
        swan_spec["ywnd"] = swan_spec["ywnd"].fillna(0)

        avg_wind = np.nanmean(
            (swan_spec.xwnd.values[:, 0] ** 2 + swan_spec.ywnd.values[:, 0] ** 2) ** 0.5
        )

        if np.abs(np.nanmean(swan_spec.depth.values) - depth) > 0.1:
            print(f"depth is not constant!")

        if np.abs(avg_wind - wind) > 0.1:
            print(f"wind speed is not constant in time!")

        # check wind from map output, check if direction is constant
        if avg_wind != 0:
            # check wind direction constant, based on x and y wind components, and nautical convention
            winddirectioncheck = (
                270
                - np.rad2deg(
                    np.arctan2(swan_spec.ywnd.values[:, 0], swan_spec.xwnd.values[:, 0])
                )
            ) % 360

            if np.abs(winddirectioncheck - wind_dir).max() > 1:
                print(f"wind direction is not constant")
                dir_trigger = False

            else:
                print(f"wind direction is constant")
                dir_trigger = True
        else:
            dir_trigger = True

        if dir_trigger:
            ds = xr.Dataset(
                {
                    "energy_density": (
                        ("time", "x-location", "frequency", "directions"),
                        E,
                    ),
                    "wind_speed": wind,
                    "wind_direction": wind_dir,
                    "distance_30min": (
                        ("time", "x-location"),
                        distance_array.T[id_wanted, :],
                    ),
                    # "water_depth": depth,
                },
                coords={
                    "time": swan_spec.time[id_wanted].values,
                    "x-location": swan_spec.x[y_index].values,
                    "y-location": output_y,
                    "frequency": swan_spec.frequency.values,
                    "directions": swan_spec.direction.values,
                },
            )
        else:
            raise ValueError("Wind direction is not constant, cannot write netCDF")
            # ds = xr.Dataset(
            #     {
            #         "energy_density": (
            #             ("time", "x-location", "frequency", "directions"),
            #             E,
            #         ),
            #         "wind_speed": wind,
            #         "wind_direction": (("time"), winddirectioncheck[id_wanted]),
            #         "distance_30min": output_x,
            #     },
            #     coords={
            #         "time": swan_spec.time[id_wanted].values,
            #         "x-location": swan_spec.x[y_index].values,
            #         "y-location": output_y,
            #         "frequency": swan_spec.frequency.values,
            #         "directions": swan_spec.direction.values,
            #     },
            # )

        ds["energy_density"].attrs["units"] = "m^2/Hz/deg"  ## todo check
        ds["energy_density"].attrs["long_name"] = "Energy Density"
        ds["wind_speed"].attrs["units"] = "m/s"
        ds["wind_direction"].attrs["units"] = "degrees from north (clockwise positive)"
        ds["distance_30min"].attrs[
            "description"
        ] = "Distance wave with peak period travels in 30 minutes based on group speed in deep water"

        ds.attrs["description"] = (
            "Energy density from SWAN calculations, per 10 minutes"
        )

        # locatie, frequentie, invoer, ... etc toevoegen

        ds.to_netcdf(
            os.path.join(fig_path, sim_name, sim_name + "_output.nc"),
            mode="w",
            engine="netcdf4",
        )

    # except Exception as e:
    #     print(f"Error processing {sim_name}: {e}")
    #     failed_sims.append((sim_name, "exception in analysis"))
    #     continue


for item in failed_sims:
    print("failed sim: {} - reason: {}".format(item[0], item[1]))
