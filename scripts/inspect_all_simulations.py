import sys
import os
from datetime import datetime, timedelta
import itertools
from tempfile import tempdir
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

fig1, ax1 = plt.subplots(figsize=[12, 12], layout="constrained")
fig2, ax2 = plt.subplots(figsize=[12, 12], layout="constrained")
fig3, ax3 = plt.subplots(figsize=[12, 12], layout="constrained")

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

    path_overview = os.path.join(fig_path, "00_overview_figs")
    ## create fig path
    if not os.path.exists(path_overview):
        os.mkdir(path_overview)

    output_x = math.floor(groupspeed(depth, offshore_peak_period) * 1800 / 1000) * 1000

    ############################################
    ## read point output
    ############################################

    tab_data = read_SWAN_tab(
        os.path.join(path_output, "POINTS_P2.TAB")
    )  # does not work perfectly

    Hsig_xoutput = tab_data.Hsig.values[
        np.where(tab_data.Xp == output_x)[0][0],
        np.where(tab_data.Yp == output_y)[0][0],
        :,
    ]
    Hsig_x0 = tab_data.Hsig.values[
        np.where(tab_data.Xp == 0)[0][0],
        np.where(tab_data.Yp == output_y)[0][0],
        :,
    ]

    ############################################
    #### Hsig vs Tp at specific location

    ax1.plot(
        Hsig_xoutput,
        tab_data.TPsmoo.values[
            np.where(tab_data.Xp == output_x)[0][0],
            np.where(tab_data.Yp == output_y)[0][0],
            :,
        ],
        ".",
    )

    ############################################
    #### Hsig at x0 versus output_x
    ax2.plot(
        Hsig_x0,
        Hsig_xoutput,
        ".",
    )

    ############################################
    #### Hsig diff (in time) at x0 versus Hsig at x0

    ax3.plot(
        Hsig_x0,
        np.diff(Hsig_x0, prepend=Hsig_x0[0]),
        ".",
    )


## save figs after loop

ax1.set_xlabel("Hsig (m)")
ax1.set_ylabel("Tp (s)")
ax1.set_title("Hsig vs Tp at x_output, y={}".format(output_y))
dir1 = os.path.join(path_overview, "Hsig_vs_Tp")
if not os.path.exists(dir1):
    os.mkdir(dir1)
fig1.savefig(
    os.path.join(
        dir1,
        "Hsig_vs_Tp_x_output_y={}.png".format(output_y),
    )
)

ax2.set_xlabel("Hsig at x=0 (m)")
ax2.set_ylabel("Hsig at x={}".format(output_x))
ax2.set_title("Hsig at x=0 versus Hsig at x_output, y={}".format(output_y))
dir2 = os.path.join(path_overview, "Hsig_x0_vs_Hsig_x")
if not os.path.exists(dir2):
    os.mkdir(dir2)
fig2.savefig(
    os.path.join(
        dir2,
        "Hsig_x0_vs_Hsig_x_output_y={}.png".format(output_y),
    )
)

ax3.set_xlabel("Hsig at x=0 (m)")
ax3.set_ylabel("dHsig/dt at x=0 (m/10min)")
ax3.set_title("Hsig change rate at x=0 versus Hsig at x=0, y={}".format(output_y))
dir3 = os.path.join(path_overview, "dHsig_dt_x0_vs_Hsig_x0")
if not os.path.exists(dir3):
    os.mkdir(dir3)
fig3.savefig(
    os.path.join(
        dir3,
        "dHsig_dt_x0_vs_Hsig_x0_y={}.png".format(output_y),
    )
)
