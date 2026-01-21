import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm
import xarray as xr
import math
import argparse
import re
from datetime import datetime
from zoneinfo import ZoneInfo


#################################
## script to analyse SWAN simulations
## Requirements:
## - swan toolbox

print("Python: current working directory:", os.getcwd())

## add swan toolbox to path
sys.path.append(os.path.abspath(os.path.join("swan_aquafind","scripts","swan-py-toolbox")))
from swanToolBox import extract_accuracy_data

from swanToolBox.swan_postprocessing.read_SWAN import (
    read_SWAN_tab,
)

# Comment this line in innit of swan_postprocessing\: from .animate_**


def git_version():
    from subprocess import Popen, PIPE

    gitproc = Popen(
        ["git", "rev-parse", "HEAD"],
        stdout=PIPE,
        cwd=os.path.join(os.getcwd(), "swan_aquafind"),
    )
    (stdout, _) = gitproc.communicate()
    return stdout.strip()


###############################################################################
## user inputs
###############################################################################

parser = argparse.ArgumentParser(description="Pass variables to Python script")
parser.add_argument("--sim_name", type=str, help="sim_name")
args = parser.parse_args()

sim_name = args.sim_name

print("analysing:", sim_name)

## path to the main folder where the simulations will be stored
model_path = "01_sims"
fig_path = ""


## settings
analyse_map = True
analyse_points = True
spec_output = True
write_netCDF = True

output_y = 15000

depth = 2000

spin_up_time = np.timedelta64(3, "h")

match = re.search(r"Tp=([\d.]+)", sim_name)
offshore_peak_period = float(match.group(1))
match2 = re.search(r"Dw=([\d.]+)", sim_name)
offshore_wave_dir = float(match2.group(1))
match3 = re.search(r"_u=([\d.]+)", sim_name)
wind = float(match3.group(1))
match4 = re.search(r"Du=([\d.]+)", sim_name)
wind_dir = float(match4.group(1))
match5 = re.search(r"Hm0=([\d.]+)", sim_name)
offshore_wave_height = float(match5.group(1))
match6 = re.search(r"Dspr=([\d.]+)", sim_name)
offshore_dspr = float(match6.group(1))


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


# Check if running in Docker (output folder exists in current directory)
IS_DOCKER = os.path.exists("output")

# Set paths based on environment
if IS_DOCKER:
    path_output = "output"
    figure_output_path = os.path.join(os.getcwd(), path_output, "figures")
    print(
        f"Running in Docker, output folder: {path_output}, figure output path: {figure_output_path}"
    )
# Running locally
elif os.path.exists(os.path.join(model_path, sim_name, "output")):
    path_output = os.path.join(model_path, sim_name, "output")
    figure_output_path = os.path.join(fig_path, sim_name)
    print(
        f"Running locally, output folder: {path_output}, figure output path: {figure_output_path}"
    )
else:
    raise FileNotFoundError(
        f"Cannot find output folder at: {os.path.join(model_path, sim_name, 'output')} or ./output"
    )

# Create figure output path if it doesn't exist
if not os.path.exists(figure_output_path):
    os.makedirs(figure_output_path, exist_ok=True)

try:
    ###############################################################################
    ## spectrum output
    ###############################################################################
    if spec_output:
        swan_spec = xr.open_dataset(
            os.path.join(path_output, "SPEC_P2_original.nc"), engine="netcdf4"
        )

        fig, ax = plt.subplots(2, 3, figsize=[15, 8], layout="constrained")
        time_index0 = 10
        time_index1 = -20
        time_index2 = -10

        loc_index1 = 0
        loc_index2 = np.where(
            (swan_spec.y.values == output_y)
            & (
                swan_spec.x.values
                == np.max(
                    swan_spec.x.values[np.where(swan_spec.y.values == output_y)[0]]
                )
            )
        )[0][0]

        dir_array = swan_spec.direction.values
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
        plt.savefig(os.path.join(figure_output_path, "spec.png"))
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
        plt.legend(loc="best")
        plt.savefig(os.path.join(figure_output_path, "bed.png"))
        plt.close(fig)
        #############
        maptime = map.time.values
        maptime_first = maptime[3]
        fig, ax = plt.subplots(figsize=[12, 12])
        pcm = ax.pcolor(
            map.x.values,
            map.y.values,
            map.hs.values[3, :, :],
            vmin=np.nanmin(map.hs.values[3, :, :]),
            vmax=np.nanmax(map.hs.values[3, :, :]),
        )
        step = 10
        plt.quiver(
            map.x.values[::step],
            map.y.values[::step],
            -1 * np.cos(np.deg2rad(map.theta0.values[3, ::step, ::step] - 90)),
            -1 * np.sin(np.deg2rad(map.theta0.values[3, ::step, ::step] - 90)),
        )
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Hs [m]")
        plt.plot(
            tab_dummy.iloc[:, 1], tab_dummy.iloc[:, 2], "k.", label="buoy locations"
        )
        plt.xlabel("RD x (m)")
        plt.ylabel("RD y (m)")
        plt.title("Hs" + str(pd.to_datetime(maptime_first)))
        plt.legend(loc="best")
        plt.axis("equal")
        plt.savefig(os.path.join(figure_output_path, "Hs_t0_map.png"))
        plt.close(fig)
        maptime_last = maptime[-1]
        fig, ax = plt.subplots(figsize=[12, 12])
        pcm = ax.pcolor(
            map.x.values,
            map.y.values,
            map.hs.values[-1, :, :],
            vmin=np.nanmin(map.hs.values[-1, :, :]),
            vmax=np.nanmax(map.hs.values[-1, :, :]),
        )
        step = 10
        plt.quiver(
            map.x.values[::step],
            map.y.values[::step],
            -1 * np.cos(np.deg2rad(map.theta0.values[-1, ::step, ::step] - 90)),
            -1 * np.sin(np.deg2rad(map.theta0.values[-1, ::step, ::step] - 90)),
        )
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Hs [m]")
        plt.plot(
            tab_dummy.iloc[:, 1], tab_dummy.iloc[:, 2], "k.", label="buoy locations"
        )
        plt.xlabel("RD x (m)")
        plt.ylabel("RD y (m)")
        plt.title("Hs" + str(pd.to_datetime(maptime_last)))
        plt.legend(loc="best")
        plt.axis("equal")
        plt.savefig(os.path.join(figure_output_path, "Hs_map.png"))
        plt.close(fig)
        #############
        maptime = map.time.values
        maptime_last = maptime[-1]
        fig, ax = plt.subplots(figsize=[12, 12])
        pcm = ax.pcolor(map.x.values, map.y.values, map.spread.values[-1, :, :])
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label(r"Dspr [$^\circ$]")
        # plt.plot(tab_dummy.iloc[:, 1], tab_dummy.iloc[:, 2], "k.", label="buoy locations")
        plt.xlabel("RD x (m)")
        plt.ylabel("RD y (m)")
        plt.title("Dspr " + str(pd.to_datetime(maptime_last)))
        plt.legend(loc="best")
        plt.axis("equal")
        plt.savefig(os.path.join(figure_output_path, "Dspr_map.png"))
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
    plt.legend(loc="best")
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, iterations, marker="o")
    plt.ylabel("Iterations")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(figure_output_path, "accur.png"))
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
        ax.legend(loc="best")

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
        fig.savefig(os.path.join(figure_output_path, "point_output.png"))
        plt.close(fig)

    #################################################
    ## quality checks
    #################################################

    ########## SWAN warnings ######################################################
    SWAN_warning_message = []
    # open the PRINT text file and check for warnings
    with open(os.path.join(path_output, "..", "PRINT"), "r") as f:
        print_contents = f.read()
        lower = print_contents.lower()
        if ("warning" in lower) or ("error" in lower):
            lines = lower.splitlines()
            ## add the warning lines to error message
            warning_lines = [line for line in lines if "warning" in line]
            error_lines = [line for line in lines if "error" in line]
            SWAN_warning_message.extend(warning_lines)
            SWAN_warning_message.extend(error_lines)

    ###############################################################################
    passed_quality_checks = True
    error_message = []

    ########## checks voor evidente fouten
    # check depth
    depth_tab = tab_data.Depth.values
    if np.abs(depth - depth_tab).max() > 0.001:
        print(f"depth deviates more than 0.001m from expected depth")
        passed_quality_checks = False
        error_message.append("depth deviation")
    # check Hm0 based on limits
    Hm0 = tab_data.Hsig.values
    if (Hm0 < 0).any() or (Hm0 > 6).any():
        print(f"Hm0 has values outside acceptable range (0-6m)")
        passed_quality_checks = False
        error_message.append("Hm0 out of range")
    # check Tp based on limits
    Tp = tab_data.TPsmoo.values[:, :, 1:]
    if (Tp < 0.5).any() or (Tp > 30).any():
        print(f"Tp has values outside acceptable range (0-30s)")
        passed_quality_checks = False
        error_message.append("Tp out of range")
    # check Dspr based on limits
    Dspr = tab_data.Dspr.values[:, :, 1:]
    if (Dspr < 5).any() or (Dspr > 100).any():
        print(f"Dspr has values outside acceptable range (5-100 degrees)")
        passed_quality_checks = False
        error_message.append("Dspr out of range")

    ############ check based on Hm0, Tp during spinuptime
    id_wanted = (swan_spec.time.values - swan_spec.time[0].values) <= spin_up_time

    Hm0_x0 = tab_data.Hsig.values[
        np.where(tab_data.Xp == 0)[0][0],
        np.where(tab_data.Yp == output_y)[0][0],
        id_wanted,
    ]
    Hm0_outputx = tab_data.Hsig.values[
        np.where(tab_data.Xp == output_x)[0][0],
        np.where(tab_data.Yp == output_y)[0][0],
        id_wanted,
    ]
    if (np.abs(Hm0_outputx / Hm0_x0)).min() < 0.9:
        print(f"Hm0 at output x is lower than 90% from Hm0 at x=0 during spinup time")
        passed_quality_checks = False
        error_message.append("Hm0 deviation during spinup")

    Tp_x0 = tab_data.TPsmoo.values[
        np.where(tab_data.Xp == 0)[0][0],
        np.where(tab_data.Yp == output_y)[0][0],
        id_wanted,
    ]
    Tp_outputx = tab_data.TPsmoo.values[
        np.where(tab_data.Xp == output_x)[0][0],
        np.where(tab_data.Yp == output_y)[0][0],
        id_wanted,
    ]

    if (np.abs(Tp_x0 / Tp_outputx - 1)).max() > 0.1:
        print(
            f"Tp at output x deviates more than 10% from Tp at x=0 during spinup time"
        )
        passed_quality_checks = False
        error_message.append("Tp deviation during spinup")

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
        passed_quality_checks = False
        error_message.append("depth not constant")

    if np.abs(avg_wind - wind) > 0.1:
        print(f"wind speed is not constant in time!")
        passed_quality_checks = False
        error_message.append("wind speed not constant")

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
            passed_quality_checks = False
            error_message.append("wind direction not constant")

        else:
            print(f"wind direction is constant")
            dir_trigger = True
    else:
        dir_trigger = True

    #################################################
    ## netcdf
    #################################################
    if spec_output and write_netCDF:

        if dir_trigger:
            ds = xr.Dataset(
                {
                    "energy_density": (
                        ("time", "x_location", "frequency", "directions"),
                        E,
                    ),
                    "wind_speed": (
                        ("time", "x_location"),
                        wind * np.ones_like(distance_array.T[id_wanted, :]),
                    ),
                    "wind_direction": (
                        ("time", "x_location"),
                        wind_dir * np.ones_like(distance_array.T[id_wanted, :]),
                    ),
                    "mean_offshore_wave_height": offshore_wave_height,
                    "mean_offshore_peak_period": offshore_peak_period,
                    "mean_offshore_wave_direction": offshore_wave_dir,
                    "mean_offshore_dspr": offshore_dspr,
                    "water_depth": (
                        ("x_location"),
                        depth * np.ones(len(swan_spec.x[y_index])),
                    ),
                    "distance_30min": (
                        ("time", "x_location"),
                        distance_array.T[id_wanted, :],
                    ),
                    "quality_check_passed": passed_quality_checks,
                    "error_message": error_message,
                    "SWAN_warnings": SWAN_warning_message,
                    "local_time_of_analysis": str(
                        datetime.now(ZoneInfo("Europe/Amsterdam")).strftime(
                            "%Y-%m-%d %H:%M:%S %Z%z"
                        )
                    ),
                },
                coords={
                    "time": swan_spec.time[id_wanted].values,
                    "x_location": swan_spec.x[y_index].values,
                    "y_location": output_y,
                    "frequency": swan_spec.frequency.values,
                    "directions": swan_spec.direction.values,
                },
            )
        else:
            raise ValueError("Wind direction is not constant, cannot write netCDF")
            # ds = xr.Dataset(
            #     {
            #         "energy_density": (
            #             ("time", "x_location", "frequency", "directions"),
            #             E,
            #         ),
            #         "wind_speed": wind,
            #         "wind_direction": (("time"), winddirectioncheck[id_wanted]),
            #         "distance_30min": output_x,
            #     },
            #     coords={
            #         "time": swan_spec.time[id_wanted].values,
            #         "x_location": swan_spec.x[y_index].values,
            #         "y_location": output_y,
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
        git_revision = git_version()
        ds.attrs["git_revision"] = "git_revision_hash: {}".format(
            git_revision.decode("utf-8")
        )

        # locatie, frequentie, invoer, ... etc toevoegen

        ds.to_netcdf(
            os.path.join(path_output, "SPEC_P2.nc"),
            mode="w",
            engine="netcdf4",
        )


except Exception as e:
    print(f"Error processing {sim_name}: {e}")
