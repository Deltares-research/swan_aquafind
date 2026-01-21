import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import glob
from mako.template import Template
from netCDF4 import Dataset, num2date, date2num
from scipy import interpolate
import logging
from pyproj import Transformer  # Added!
import re



logging.basicConfig(level=logging.INFO)


def extract_accuracy_data(file_path):
    # Read the content of the file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Initialize lists to store time and accuracy values

    time_steps = []
    accuracy_values = []
    current_time = None
    current_accuracies = []

    # Process each line
    for line in lines:
        # Extract time of computation
        time_match = re.search(r"Time of computation\s*->\s*(\d+\.\d+)", line)
        if time_match:
            if current_time and current_accuracies:
                time_steps.append(current_time)
                accuracy_values.append(current_accuracies)
            current_time = time_match.group(1)
            current_accuracies = []

        # Extract accuracy values
        acc_match = re.search(r"accuracy OK in\s*([\d.]+)\s*%", line)
        if acc_match:
            current_accuracies.append(float(acc_match.group(1)))

    # Prepare data for plotting
    x = list(range(len(time_steps)))
    accur_mean = [sum(acc) / len(acc) for acc in accuracy_values]
    accur_max = [max(acc) for acc in accuracy_values]

    return time_steps, accur_max, accur_mean, accuracy_values


def create_pseudowind(taux, tauy, pseudowind_file):
    """create pseudowind based on windstress

    Args:
        taux (array): 2D array with the x component of the shear stress
        tauy (araay): 2D array with the y component of the shear stress
        pseudowind_file (string): path to the file with the look-up table

    Returns:
        array: 2D array with u and v wind speed
    """

    tau_abs = np.sqrt(taux**2 + tauy**2)

    lookup = np.loadtxt(pseudowind_file)

    Uspeed_abs = np.interp(tau_abs, lookup[:, 0], lookup[:, 1])

    U = (taux * Uspeed_abs) / tau_abs
    V = (tauy * Uspeed_abs) / tau_abs
    return U, V


def write_sp2(file, dates, energy, lon, lat, freq, dir, lonlat=True):
    """create sp2 file

    Args:
        file (string): filename
        dates (list): list with datetimes
        energy (array): 4D array with the energy density with the folwing dimensions: dates, directions, frequencies and locations.
        lon (array): 1D array with coordinates (longitude)
        lat (array): 1D array with coordinates (latitude)
        freq (array): 1D array with frequencies
        dir (array): 1D array with directions
    """

    with open(file, "w") as f:
        f.writelines("SWAN\n")
        f.writelines(
            "$ Data produced by Python on {} \n".format(
                datetime.now().strftime("%Y%m%d%H%M")
            )
        )

        f.writelines("TIME                                 time-dependent data\n")
        f.writelines("1                                    time coding option\n")
        if lonlat == True:
            f.writelines(
                "LONLAT                               locations in spherical coordinates\n"
            )
        else:
            f.writelines(
                "LOCATIONS                               locations in x-y-space\n"
            )

        f.writelines(
            "{:d}                                 number of locations \n".format(
                len(lon)
            )
        )
        for i in range(len(lon)):
            f.write("{:f} {:f}\n".format(lon[i], lat[i]))
        nxy = len(lon)

        f.writelines(
            "AFREQ                                absolute frequencies in Hz\n"
        )
        f.writelines(
            "{:d}                                 number of frequencies \n".format(
                len(freq)
            )
        )
        for item in freq:
            f.writelines("{:g}\n".format(item))

        f.writelines(
            "NDIR                                 spectral nautical directions in degr\n"
        )
        f.writelines(
            "{:d}                                 number of directions \n".format(
                len(dir)
            )
        )
        for d in dir:
            f.writelines("{:g}\n".format(d))

        # QUANT
        f.writelines("QUANT \n")
        f.writelines(
            "1                                       number of quantities in table\n"
        )
        f.writelines(
            "VaDens                                  variance densities in m2/Hz/degr\n"
        )
        f.writelines("m2/Hz/degr                              unit\n")
        f.writelines("-99                                     exception value" + "\n")

        if isinstance(dates[0], np.datetime64):
            for ii_time, date in enumerate(dates):
                f.writelines(
                    "{}                         date and time\n".format(
                        date.astype("M8[ms]").astype(datetime).strftime("%Y%m%d.%H%M%S")
                    )
                )
                for ii_loc in range(nxy):
                    f.writelines("FACTOR \n1\n")
                    for ii_freq, ff in enumerate(freq):
                        for ii_dir, d in enumerate(dir):
                            f.writelines(
                                "{:g} ".format(energy[ii_time, ii_dir, ii_freq, ii_loc])
                            )
                        f.writelines("\n")
        elif isinstance(dates[0], datetime):
            for ii_time, date in enumerate(dates):
                f.writelines(
                    "{}                         date and time\n".format(
                        date.strftime("%Y%m%d.%H%M%S")
                    )
                )
                for ii_loc in range(nxy):
                    f.writelines("FACTOR \n1\n")
                    for ii_freq, ff in enumerate(freq):
                        for ii_dir, d in enumerate(dir):
                            f.writelines(
                                "{:g} ".format(energy[ii_time, ii_dir, ii_freq, ii_loc])
                            )
                        f.writelines("\n")




def create_wind_nc(nc_path, fname, xgrid, ygrid, uwind, vwind, dates, fig=True):

    ## create netCDF
    nc_wind = Dataset(
        os.path.join(nc_path, fname + "_wind.nc"), "w", format="NETCDF3_CLASSIC"
    )

    ## create dimensions
    nc_wind.createDimension("time", len(dates))
    nc_wind.createDimension("y", ygrid.shape[0])
    nc_wind.createDimension("x", xgrid.shape[1])

    ## Create variables, time
    time = nc_wind.createVariable("time", np.float64, ("time",))
    time.units = "minutes since 1970-01-01 00:00:00.0 +0000"
    time.long_name = "time"
    time.standard_name = "time"
    time.axis = "T"
    ## variables, y
    y = nc_wind.createVariable("y", np.float64, ("y",), fill_value=9.96921e36)
    y.units = "m"
    y.long_name = "y coordinate"
    y.standard_name = "projection_y_coordinate"
    y.axis = "Y"
    ## variables, x
    x = nc_wind.createVariable("x", np.float64, ("x",), fill_value=9.96921e36)
    x.units = "m"
    x.long_name = "x coordinate"
    x.standard_name = "projection_x_coordinate"
    x.axis = "X"
    ## variables, z
    z = nc_wind.createVariable("z", np.float64, ("y", "x"), fill_value=9.96921e36)
    z.units = "meters"
    z.long_name = "height above mean sea level"
    ## variables, lat
    lat = nc_wind.createVariable("lat", np.float64, ("y", "x"), fill_value=9.96921e36)
    lat.units = "degrees_north"
    z.long_name = "latitude"
    ## variables, lon
    lon = nc_wind.createVariable("lon", np.float64, ("y", "x"), fill_value=9.96921e36)
    lon.units = "degrees_east"
    z.long_name = "longitude"
    ## variables, crs
    crs = nc_wind.createVariable("crs", np.int32)
    crs.long_name = "coordinate reference system"
    # crs.crs_wkt = f"{transformer.target_crs}"  ## ???
    ## variables, ux
    ux = nc_wind.createVariable(
        "eastward_wind", np.float32, ("time", "y", "x"), fill_value=-999.0
    )
    ux.units = "m/s"
    ux.long_name = "eastward_wind"
    ux.standard_name = "eastward_wind"
    ux.grid_mapping = "crs"
    ## variables, uy
    uy = nc_wind.createVariable(
        "northward_wind", np.float32, ("time", "y", "x"), fill_value=-999.0
    )
    uy.units = "m/s"
    uy.long_name = "northward_wind"
    uy.standard_name = "northward_wind"
    uy.grid_mapping = "crs"

    ## write data
    logging.info("Start writing input netCDF")

    nc_wind.variables["y"][:] = ygrid[:, 0]
    nc_wind.variables["x"][:] = xgrid[0, :]
    # temp_time_index = 0
    Nt = uwind.shape[0]
    for jj in range(Nt):

        # Define time
        time[jj] = date2num(
            dates[jj],
            units="minutes since 1970-01-01 00:00:00.0 +0000",
        )

        nc_wind.variables["eastward_wind"][jj, :, :] = uwind[jj, :, :]
        nc_wind.variables["northward_wind"][jj, :, :] = vwind[jj, :, :]

    nc_wind.close()

    if fig:
        logging.info("generate figures")
        ##
        nc_fig = Dataset(os.path.join(nc_path, fname + "_wind.nc"), "r")
        X = nc_fig.variables["x"][:]
        Y = nc_fig.variables["y"][:]
        times = nc_fig.variables["time"]

        dates = np.empty(len(times[:]), dtype="datetime64[m]")
        for ii, item_t in enumerate(times[:]):
            dates[ii] = num2date(int(item_t), units=time.units)

        for ii, date in enumerate(dates):
            ux = nc_fig.variables["eastward_wind"][ii, :, :]
            uy = nc_fig.variables["northward_wind"][ii, :, :]

            u_abs = np.sqrt(ux**2 + uy**2)
            plt.figure(figsize=[10, 10])
            plt.pcolor(X, Y, u_abs)
            plt.colorbar()
            plt.quiver(X[::3], Y[::3], ux[::3, ::3], uy[::3, ::3])
            plt.axis("equal")
            plt.title(date)
            plt.xlabel("x RD new [m]")
            plt.ylabel("y RD new [m]")

            if not os.path.exists(os.path.join(nc_path, fname)):
                os.mkdir(os.path.join(nc_path, fname))

            plt.savefig(
                os.path.join(nc_path, fname.replace(".nc", ""), "{}.png".format(ii))
            )
            plt.close()
        nc_fig.close()
    logging.info("Finished")






class SWANModelSetup:
    """Class to generate SWAN models"""

    def __init__(self, fname, template_path=None):
        """innit

        Args:
            fname (string): name of the simulation
            template_path (string, optional): string with location to template files Defaults to None.
        """
        self.fname = fname
        if template_path == None:
            self.templatePath = os.path.join(
                os.path.abspath(os.path.dirname(__file__)), "template"
            )
        else:
            self.templatePath = os.path.join(os.path.abspath(template_path))

        ## default cluster setting
        self.cluster = {"partition": "4vcpu", "runtime": "1-00:00:00"}

        ## default settings
        self.settings = {
            "mode": "NONST",
            "cdcap": 0.00275,
            "level": 0,
            "swan_module": "41.31A.1_intel18.0.3",
            "swan_exe": "swan_4131A_1_del_l64_i18_omp.exe",
            "coordinates": "SPHERICAL",
            "start": None,
            "dt": 60,
            "stop": None,
            "stat": False,
            "NUMSCHEME": "PROP BSBT",
            "NUM": None,
            "dt_unit": "MIN",
            "meta_string": "",
        }

        self.physics = {
            "QUAD": None,
            "TRIAD": None,
            "FRIC": None,
            "BREA": None,
            "winddrag": None,
            "SSWELL": None,
            "WCAP": None,
            "GEN3": None,
        }

        self.bottom = {
            "bottom_info": None,
            "obstacles_path": None,
            "gridType_bot": None,
            "xpinp": 0,
            "ypinp": 0,
            "alpinp": 0,
            "mxinp": None,
            "myinp": None,
            "dxinp": None,
            "dyinp": None,
        }

        self.grid = {
            "dir_res": 72,
            "gridType": "CURV",
            "xpc": 0,
            "ypc": 0,
            "alpc": 0,
            "xlenc": None,
            "ylenc": None,
            "flow": 0.03,
            "fhigh": 1,
            "Nx": None,
            "Ny": None,
            "grid_info": None,
        }

        self.data = {
            "water_level": False,
            "currents": False,
            "wind": -1,
            "pnt": {},
            "fname": fname,
            "block_variables": [
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
            "bc_type": None,
            "bc_path": None,
        }

    def __repr__(self):
        return self.fname

    def set_cluster(self, partition=None, runtime=None):
        """set the cluster settings (h7, omp only)

        Args:
            partition (str): string to indicate vcpu partitioning (only omp)
            runtime (datetime.timedelta): maximum runtime of job on h7
        """
        self.cluster["partition"] = partition

        # Extract components runtime
        total_seconds = int(runtime.total_seconds())
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        # Format as "dd-HH:MM:SS"
        self.cluster["runtime"] = f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"

    def set_settings(self, input_struct):
        """set the settings

        Args:
            input_struct (dict): dictionairy with input settings
        """
        if "stat" in input_struct:
            if input_struct["stat"]:
                self.settings["stat_dates"] = []
                Ncomp = (input_struct["stop"] - input_struct["start"]) / timedelta(
                    minutes=input_struct["dt"]
                )
                for ii in range(int(Ncomp)):
                    self.settings["stat_dates"].append(
                        (
                            input_struct["start"]
                            + ii * timedelta(minutes=input_struct["dt"])
                        ).strftime("%Y%m%d.%H%M")
                    )
        ## set settings
        for item in self.settings:
            if item in input_struct:
                if item in ["start", "stop"]:
                    self.settings[item] = input_struct[item].strftime("%Y%m%d.%H%M")
                else:
                    self.settings[item] = input_struct[item]
                ## remove from input_dict
                input_struct.pop(item)

        if not bool(input_struct):
            for item in input_struct:
                print("{} not added. Is not an input parameter".format(item))

    def set_grid(self, gridType, input_struct):
        """set grid related settings

        Args:
            input_struct (dict): dictionairy with grid related settings
        """
        if gridType == "REGULAR":
            self.grid["gridType"] = gridType
        elif gridType == "CURV":
            self.grid["gridType"] = gridType
        elif gridType == "UNSTR":
            self.grid["gridType"] = gridType
        else:
            print("grid Type does not exist")
        ## set grid
        for item in self.grid:
            if item in input_struct:
                self.grid[item] = input_struct[item]
                ## remove from input_dict
                input_struct.pop(item)
        if not bool(input_struct):
            for item in input_struct:
                print("{} not added. Is not an input parameter".format(item))

    def set_bottom(self, gridType, input_struct):
        """set bottom related settings

        Args:
            gridType (str): string with REGULAR or CURV
            input_struct (struct): struct with input data
        """

        if gridType == "REGULAR":
            self.bottom["gridType_bot"] = gridType
        elif gridType == "CURV":
            self.bottom["gridType_bot"] = gridType
        elif gridType == "UNSTR":
            self.bottom["gridType_bot"] = gridType
        else:
            print("grid Type does not exist")
        ## set bottom
        for item in self.bottom:
            if item in input_struct:
                self.bottom[item] = input_struct[item]
                ## remove from input_dict
                input_struct.pop(item)
        if not bool(input_struct):
            for item in input_struct:
                print("{} not added. Is not an input parameter".format(item))

    def set_physics(self, input_struct):
        """set physics

        Args:
            input_struct (dict): dictionairy with physics related settings
        """
        ## set
        for item in self.physics:
            if item in input_struct:
                self.physics[item] = input_struct[item]
                ## remove from input_dict
                input_struct.pop(item)
        if not bool(input_struct):
            for item in input_struct:
                print("{} not added. Is not an input parameter".format(item))

    def set_water_level(self, water_level_path, start, dt, stop):
        """set water level

        Args:
            water_level_path (string): path to netCDEF with water levels
            start (datetime): datetime object with start date
            dt (int): integer with timestep in minutes
            stop (datetime): datetime object with stop date
        """

        self.data["water_level"] = True
        self.data["water_level_path"] = water_level_path.replace("\\", "/")
        self.data["water_level_start"] = start.strftime("%Y%m%d.%H%M")
        self.data["water_level_dt"] = dt
        self.data["water_level_stop"] = stop.strftime("%Y%m%d.%H%M")

    def set_flow(self, currents_path, start=None, dt=None, stop=None, currents_nc=True):
        """set flow

        Args:
            currents_path (string): path to netCDEF with flow
            start (datetime): datetime object with start date
            dt (int): integer with timestep in minutes
            stop (datetime): datetime object with stop date
        """
        self.data["currents"] = True
        self.data["currents_path"] = currents_path.replace("\\", "/")
        if currents_nc:
            self.data["currents_start"] = start.strftime("%Y%m%d.%H%M")
            self.data["currents_dt"] = dt
            self.data["currents_stop"] = stop.strftime("%Y%m%d.%H%M")
        self.data["currents_nc"] = currents_nc

    def set_wind(
        self,
        type="constant",
        wind_path=None,
        start=None,
        dt=None,
        stop=None,
        wind_speed=None,
        wind_dir=None,
        wind_factor=1,
    ):
        """set wind

        Args:
            type (str, optional): type of wind forcing [constant or field]. Defaults to 'constant'.
            wind_path (str, optional): path to netCDF file with wind. Defaults to None.
            start (datetime, optional): dattime with start date. Defaults to None.
            dt (int, optional): integer with time step in minutes. Defaults to None.
            stop (datetime, optional): datetime with stop date. Defaults to None.
            wind_speed (float, optional): constant wind speed in case of constant wind. Defaults to None.
            wind_dir (float, optional): constant wind direction in case of constant wind. Defaults to None.
        """
        if type == "constant":
            self.data["wind"] = 1
            self.data["wind_speed"] = wind_speed
            self.data["wind_dir"] = wind_dir
        elif type == "field":
            self.data["wind"] = 0
            self.data["wind_path"] = wind_path.replace("\\", "/")
            self.data["wind_start"] = start.strftime("%Y%m%d.%H%M")
            self.data["wind_dt"] = dt
            self.data["wind_stop"] = stop.strftime("%Y%m%d.%H%M")
            self.data["wind_factor"] = wind_factor
        else:
            print("wrong input type. Options are: constant or field")

    def set_bc(
        self,
        type=None,
        bc_path=None,
        boundary_side=None,
        bc_file="wave.par",
        initial=False,
        dspr_setting=None,
        Hm0_series=None,
        date_series=None,
        Tp_series=None,
        Dir_series=None,
        Dspr_series=None,
    ):
        """set boundary conditions

        Args:
            type (str): type of boundary conditions. Defaults to None.
            bc_path (str): path to file with boundary conditions. Defaults to None.
            boundary_side (str, optional): side to generate boundaries, for type='specsidefile'.
        """
        self.data["bc_type"] = type

        if type == "nest":
            self.data["bc_type"] = "nest"
            self.data["bc_path"] = bc_path.replace("\\", "/")
        elif type == "specsidefile":
            self.data
            self.data["boundary_side"] = boundary_side
            self.data["bc_path"] = bc_path.replace("\\", "/")
        elif type == "par":
            self.data["bc_type"] = "par"
            self.data["boundary_side"] = boundary_side
            self.data["bc_file"] = os.path.basename(
                bc_file
            )  # only works if file is in the same folder as the model setup
            if initial == False:
                self.data["initial"] = ""
            elif initial == True:
                self.data["initial"] = (
                    f"INITIAL PAR {Hm0_series[0]:.3f} {Tp_series[0]:.3f} {Dir_series[0]:.1f} {Dspr_series[0]:.1f}"
                )
            self.data["dspr_setting"] = dspr_setting
            if dspr_setting is None:
                dspr_setting = "DEGREES"
            if Dspr_series is None:
                Dspr_series = np.zones_like(Hm0_series) * 30
            with open(bc_file, "w") as f:
                f.write("TPAR\n")
                for ii in range(len(Hm0_series)):
                    f.write(
                        "{} {} {} {} {}\n".format(
                            date_series[ii].strftime("%Y%m%d.%H%M%S"),
                            Hm0_series[ii],
                            Tp_series[ii],
                            Dir_series[ii],
                            Dspr_series[ii],
                        )
                    )

        else:
            print("wrong input type. Options are: nest or specsidefile")

    def set_output_pnt(
        self, name, variables, dt, file, spec1D=True, spec2D=False, ext="nc"
    ):
        """set point output

        Args:
            name (str): string describing the output name
            variables (list): list with variables
            dt (datetime.timedelta): time step for the output
            file (str): string with the file name
            spec1D (bool, optional): boolean which indicates whether 1D spectra out put is written. Defaults to True.
            spec2D (bool, optional): boolean which indicates whether 2D spectra out put is written. Defaults to False.
        """

        if not name in self.data["pnt"]:
            self.data["pnt"][name] = {}
        self.data["pnt"][name]["pnt_variables"] = variables
        self.data["pnt"][name]["pnt_file"] = file.replace("\\", "/")
        self.data["pnt"][name]["spec1D"] = spec1D
        self.data["pnt"][name]["spec2D"] = spec2D
        self.data["pnt"][name]["ext"] = ext

        pntday = dt.total_seconds() // (24 * 3600)
        pnthr = dt.total_seconds() // (3600)
        pntmin = dt.total_seconds() // (60)

        if pntday > 0:
            output_frequency_PNT_str = f"{dt.total_seconds() / (24*3600)} DAY"
        elif pnthr > 0:
            output_frequency_PNT_str = f"{dt.total_seconds() / (3600)} HR"
        elif pntmin > 0:
            output_frequency_PNT_str = f"{dt.total_seconds() / (60)} MIN"

        self.data["pnt"][name]["pnt_dt"] = output_frequency_PNT_str

    def set_output_block(self, variables, dt=timedelta):
        self.data["block_variables"] = variables

        blockday = dt.total_seconds() // (24 * 3600)
        blockhr = dt.total_seconds() // (3600)
        blockmin = dt.total_seconds() // (60)
        if blockday > 0:
            output_frequency_block_str = f"{dt.total_seconds() / (24*3600)} DAY"
        elif blockhr > 0:
            output_frequency_block_str = f"{dt.total_seconds() / (3600)} HR"
        elif blockmin > 0:
            output_frequency_block_str = f"{dt.total_seconds() / (60)} MIN"

        self.data["block_dt"] = output_frequency_block_str

    def create_model(self, path):
        """create model at given location

        Args:
            path (string): string with path of the location where model setup should be written
        """
        ## checks
        assert self.grid["Nx"] is not None, "Nx is not specified"
        assert self.grid["Ny"] is not None, "Ny is not specified"
        assert self.settings["start"] is not None, "start is not specified"
        assert self.settings["stop"] is not None, "stop is not specified"
        ##

        self.model_path = os.path.join(path, self.fname)

        ## merge structs
        data = dict(self.data)

        for item in self.cluster:
            data[item] = self.cluster[item]
        for item in self.settings:
            data[item] = self.settings[item]
        for item in self.grid:
            data[item] = self.grid[item]
        for item in self.physics:
            data[item] = self.physics[item]
        for item in self.bottom:
            data[item] = self.bottom[item]

        self._render(os.path.join(path, self.fname), data)

    def _render(self, path, data):
        """render files

        Args:
            path (string): location where files are rendered
            data (dict): dictionairy with mako keywords
        """

        ## create path
        if not os.path.exists(path):
            os.mkdir(path)
        ## get all template files
        files = glob.glob(os.path.join(self.templatePath, "**", "*"), recursive=True)
        ## loop over files
        for file in files:
            # print(file)
            output_path = file.replace(os.path.join(self.templatePath), path)
            ## skip if is dir
            if os.path.isdir(file):
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
            else:
                with open(file) as f:
                    template = f.read()

                tmpl = Template(template)

                ## rename .swn file
                if os.path.splitext(output_path)[1] == ".swn":
                    path_tmp, filename_tmp = os.path.split(output_path)

                    output_path = os.path.join(path_tmp, "{}.swn".format(data["fname"]))

                with open(output_path, mode="w", newline="\n") as f:
                    f.write(tmpl.render(**data))
