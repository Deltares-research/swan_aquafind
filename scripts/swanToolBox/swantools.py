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

from swanToolBox.matroos import (
    get_anal_time,
    download_matroos,
    download_matroos_getmatroos,
)

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


def get_hydro(
    start,
    end,
    N_t,
    nc_path,
    fname,
    x_int,
    y_int,
    source,
    download=True,
    username="deltares",
    password="...",
    flow=True,
    fig=True,
    fm_hydro=False,
    dowloadAnalysis=True,
):
    """download data from matroos and generate input flow and water level netCDF

    Args:
        start (datetime): starttime
        end (datetime): endtime
        N_t (int): number of slices which are used from each netCDF
        nc_path (string): path of input netCDF without extension
        fname (string): name of file
        x_int (array): 1D array with x coordinated of input netCDF
        y_int (array): 1D array with y coordinated of input netCDF
        wind_source (string,): matroos source file.
        download (bool, optional): download data. Defaults to True.
        username (str, optional): matroos username. Defaults to 'deltares'.
        password (str, optional): matroos password. Defaults to '...'.
        flow (bool, optional): boolean indicating whether flow input netCDf must be created
        fig (bool, optional): create figures
    """

    with open(os.path.join(nc_path, fname + ".log"), "wb") as f:
        f.write("log {}\n".format(fname).encode("ascii"))

    ## get nc
    if dowloadAnalysis:
        nc_dates = get_anal_time(start, end, databse="maps2d", source=source)

        assert len(nc_dates) > 0, "No data available"

    with open(os.path.join(nc_path, fname + ".log"), "a") as f:
        f.write("analysis times:\n")
        for item in nc_dates:
            f.write("{}\n".format(item))
        f.write(" \n")

    ## download data
    if download:
        logging.info("start downloading netCDF")
        if dowloadAnalysis:
            download_matroos(
                "data_hydro",
                nc_dates,
                source=source,
                username=username,
                password=password,
                database="maps2d",
                hindcast=1,
            )
        else:
            start0 = start.replace(second=0, hour=0, minute=0)
            end0 = end + (datetime.min - end) % timedelta(hours=12)

            nc_dates2 = np.arange(start0, end0, timedelta(hours=12)).astype(datetime)

            download_matroos_getmatroos(
                "data_hydro",
                nc_dates2,
                source=source,
                username=username,
                password=password,
                database="maps2d",
                duration=timedelta(hours=12),
            )

            nc_dates = []
            for item in nc_dates2:
                nc_dates.append(item.strftime("%Y%m%d%H%M"))

    ##

    ## dates
    dates = np.arange(start, end, timedelta(hours=1)).astype(np.datetime64)

    nc_dates_exists = []
    indices_per_netcdf = []
    for date in nc_dates:
        nc = Dataset(os.path.join("data_hydro", source + "_" + date + ".nc"), "r")
        date = datetime.strptime(date, "%Y%m%d%H%M")

        time_nc = nc.variables["time"]
        dates_nc = np.empty(len(time_nc[:]), dtype="datetime64[m]")
        for ii, item_t in enumerate(time_nc[:]):
            dates_nc[ii] = num2date(int(item_t), units=time_nc.units)
        index = np.where(dates_nc <= date)[0]
        nc.close()
        if len(index) == 0:
            index = -1
        indices_per_netcdf.append(index)
        ## append dates
        for item in dates_nc[index]:
            nc_dates_exists.append(item)

    dates = np.asarray(nc_dates_exists)
    dates = np.unique(dates)

    ## create nc for waterlevel and flow
    for ll in range(2):
        ## get name
        if ll == 0:
            logging.info("create input water level")
            name = "_water_level.nc"
            with open(os.path.join(nc_path, fname + ".log"), "a") as f:
                f.write("processing water level\n")
        else:
            ## stop is flow is not needed
            if not flow:
                break
            logging.info("create input flow")
            name = "_flow.nc"
            with open(os.path.join(nc_path, fname + ".log"), "a") as f:
                f.write("processing flow\n")
        ## create netCDF
        nc_wl = Dataset(
            os.path.join(nc_path, fname + name), "w", format="NETCDF3_CLASSIC"
        )

        nc_wl.createDimension("x", len(x_int))
        nc_wl.createDimension("y", len(y_int))

        nc_wl.createDimension("time", len(dates))
        nc_wl.createDimension("analysis_time", 1)

        nc_wl.Conventions = "CF-1.6,UGRID-0.9"
        nc_wl.Metadata_Conventions = "Unidata Dataset Discovery v1.0"

        time = nc_wl.createVariable("time", np.float64, ("time",))
        time.units = "minutes since 1970-01-01 00:00:00.0 +0000"
        time.long_name = "time"
        time.standard_name = "time"
        time.axis = "T"

        for ii, item_t in enumerate(dates[:]):
            time[ii] = date2num(
                item_t.item(), units="minutes since 1970-01-01 00:00:00.0 +0000"
            )

        analysis_time = nc_wl.createVariable(
            "analysis_time", np.float64, ("analysis_time",)
        )
        analysis_time.units = "minutes since 1970-01-01 00:00:00.0 +0000"
        analysis_time.long_name = "forecast_reference_time"
        analysis_time.standard_name = "forecast_reference_time"

        # row = nc_wl.createVariable('row', np.float64, ('row'))
        # row.units = '1'
        # row.long_name = 'row indices'
        # row.axis = 'Y'

        # col = nc_wl.createVariable('col', np.float64, ('col'))
        # col.units = '1'
        # col.long_name = 'row indices'
        # col.axis = 'X'

        y = nc_wl.createVariable("y", np.float64, ("y"), fill_value=9.96921e36)
        y.units = "degrees_north"
        y.long_name = "y coordinate according to WGS 1984"
        y.standard_name = "latitude"
        y.axis = "Y"

        x = nc_wl.createVariable("x", np.float64, ("x"), fill_value=9.96921e36)
        x.units = "degrees_east"
        x.long_name = "x coordinate according to WGS 1984"
        x.standard_name = "longitude"
        y.axis = "X"

        crs = nc_wl.createVariable("crs", np.int32)
        # crs.units = 'degrees_north'
        crs.long_name = "coordinate reference system"
        crs.grid_mapping_name = "latitude_longitude"
        crs.longitude_of_prime_meridian = 0.0
        crs.semi_major_axis = 6378137.0
        crs.inverse_flattening = 298.257223563
        crs.epsg_code = "EPSG:4326"
        crs.proj4_params = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
        crs.crs_wkt = """GEOGCS["WGS 84",
            DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],
            PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],
            AUTHORITY["EPSG","4326"]] """

        z = nc_wl.createVariable("z", np.float64, ("x", "y"), fill_value=9.96921e36)
        z.units = "meters"
        z.long_name = "height above mean sea level"
        ## create waterlevel or flow
        if ll == 0:
            sep = nc_wl.createVariable(
                "sep", np.float32, ("time", "y", "x"), fill_value=-999.0
            )
            sep.units = "m"
            sep.long_name = "sep"
            sep.standard_name = "sea_surface_height_above_sea_level"
            sep.grid_mapping = "crs"

            sep.coordinates = "lat lon analysis_time"
        else:
            ux = nc_wl.createVariable(
                "x_sea_water_velocity",
                np.float32,
                ("time", "y", "x"),
                fill_value=-999.0,
            )
            ux.units = "m/s"
            ux.long_name = "x_sea_water_velocity"
            ux.standard_name = "eastward_sea_water_velocity"
            ux.grid_mapping = "crs"

            ux.coordinates = "lat lon analysis_time"

            uy = nc_wl.createVariable(
                "y_sea_water_velocity",
                np.float32,
                ("time", "y", "x"),
                fill_value=-999.0,
            )
            uy.units = "m/s"
            uy.long_name = "y_sea_water_velocity"
            uy.standard_name = "northward_sea_water_velocity"
            uy.grid_mapping = "crs"

            uy.coordinates = "lat lon analysis_time"

        logging.info("Write hydro input netCDF")
        for jj, item in enumerate(nc_dates):
            with open(os.path.join(nc_path, fname + ".log"), "a") as f:
                f.write("\nfile {}: ".format(item))

            nc = Dataset(os.path.join("data_hydro", source + "_" + item + ".nc"))

            ## get dates
            time_dummy = nc.variables["time"]
            date_dummy = np.empty(len(time_dummy[:]), dtype="datetime64[m]")
            for ii, item_t in enumerate(time_dummy[:]):
                dummy = num2date(int(item_t), units=time_dummy.units)
                date_dummy[ii] = date2num(dummy, units=time.units)

            ## set coords
            if jj == 0:
                nc_wl.variables["y"][:] = y_int
                nc_wl.variables["x"][:] = x_int
                if fm_hydro:
                    Mesh_face_x = nc.variables["Mesh_face_x"][:].data
                    Mesh_face_y = nc.variables["Mesh_face_y"][:].data

                else:
                    lat = nc.variables["lat"][:].data
                    lon = nc.variables["lon"][:].data

            ## set data
            for ii in range(len(indices_per_netcdf[jj])):
                index = np.where(dates == date_dummy[indices_per_netcdf[jj][ii]])[0]

                ## stop is data is not pressent
                if len(index) == 0:
                    print(item)
                    continue

                with open(os.path.join(nc_path, fname + ".log"), "a") as f:
                    f.write("{} ".format(date_dummy[ii]))

                if ll == 0:
                    [x_int_grid, y_int_grid] = np.meshgrid(x_int, y_int)
                    if fm_hydro:
                        tmp = nc.variables["sep"][indices_per_netcdf[jj][ii], :].data

                        points = np.array([Mesh_face_x, Mesh_face_y]).T

                        wlev = interpolate.griddata(
                            points, tmp, (x_int_grid, y_int_grid), method="nearest"
                        )

                        wlev[np.isnan(wlev)] = -999

                        if (wlev == -999).all():
                            print("all wlev is -999 " + source + "_" + item)

                        nc_wl.variables["sep"][index, :, :] = wlev

                    else:
                        tmp = nc.variables["sep"][indices_per_netcdf[jj][ii], :, :].data
                        tmp[tmp == -9999] = np.nan

                        ind = np.where(
                            np.logical_and(lon.flatten() < 1000, lat.flatten() < 1000)
                        )[0]

                        points = np.array([lon.flatten()[ind], lat.flatten()[ind]]).T

                        wlev = interpolate.griddata(
                            points,
                            tmp.flatten()[ind],
                            (x_int_grid, y_int_grid),
                            method="nearest",
                        )

                        wlev[np.isnan(wlev)] = -999

                        nc_wl.variables["sep"][index, :, :] = wlev

                else:
                    [x_int_grid, y_int_grid] = np.meshgrid(x_int, y_int)
                    if fm_hydro:
                        uu = nc.variables["velu"][indices_per_netcdf[jj][ii], :].data
                        uu[uu == -9999] = np.nan

                        vv = nc.variables["velv"][indices_per_netcdf[jj][ii], :].data
                        vv[vv == -9999] = np.nan

                        points = np.array([Mesh_face_x, Mesh_face_y]).T

                        uu_int = interpolate.griddata(
                            points, uu, (x_int_grid, y_int_grid), method="nearest"
                        )
                        vv_int = interpolate.griddata(
                            points, vv, (x_int_grid, y_int_grid), method="nearest"
                        )
                    else:
                        uu = nc.variables["velu"][indices_per_netcdf[jj][ii], :, :].data
                        uu[uu == -9999] = np.nan

                        vv = nc.variables["velv"][indices_per_netcdf[jj][ii], :, :].data
                        vv[vv == -9999] = np.nan
                        # todo
                        ind = np.where(
                            np.logical_and(lon.flatten() < 1000, lat.flatten() < 1000)
                        )[0]

                        points = np.array([lon.flatten()[ind], lat.flatten()[ind]]).T

                        ## check not complettly correct
                        uu_int = interpolate.griddata(
                            points,
                            uu.flatten()[ind],
                            (x_int_grid, y_int_grid),
                            method="nearest",
                        )
                        vv_int = interpolate.griddata(
                            points,
                            vv.flatten()[ind],
                            (x_int_grid, y_int_grid),
                            method="nearest",
                        )

                    uu_int[np.isnan(uu_int)] = -999
                    vv_int[np.isnan(vv_int)] = -999

                    nc_wl.variables["x_sea_water_velocity"][index, :, :] = uu_int
                    nc_wl.variables["y_sea_water_velocity"][index, :, :] = vv_int

            nc.close()

        nc_wl.close()
        ##
        if fig:
            nc_wl = Dataset(os.path.join(nc_path, fname + name), "r")
            X = nc_wl.variables["x"][:]
            Y = nc_wl.variables["y"][:]
            times = nc_wl.variables["time"]

            dates0 = np.empty(len(times[:]), dtype="datetime64[m]")
            for ii, item_t in enumerate(times[:]):
                dates0[ii] = num2date(int(item_t), units=times.units)
            if not os.path.exists(
                os.path.join(nc_path, fname + name.replace(".nc", ""))
            ):
                os.mkdir(os.path.join(nc_path, fname + name.replace(".nc", "")))
            for ii, date in enumerate(dates0):
                if ll == 0:
                    wl = nc_wl.variables["sep"][ii, :, :]

                    plt.figure()
                    plt.pcolor(X, Y, wl)
                    plt.colorbar()
                    plt.axis("equal")
                    plt.clim([-2, 2])
                    plt.title(date)
                    plt.savefig(
                        os.path.join(
                            nc_path,
                            fname + name.replace(".nc", ""),
                            fname + name.replace(".nc", "") + "{}.png".format(ii),
                        )
                    )
                    plt.close()
                else:
                    ux = nc_wl.variables["x_sea_water_velocity"][ii, :, :]
                    uy = nc_wl.variables["y_sea_water_velocity"][ii, :, :]

                    u_abs = np.sqrt(ux**2 + uy**2)
                    plt.figure(figsize=[10, 10])
                    plt.pcolor(X, Y, u_abs)
                    plt.colorbar()
                    plt.quiver(X[::20], Y[::20], ux[::20, ::20], uy[::20, ::20])
                    plt.axis("equal")
                    plt.clim([-10, 10])
                    plt.title(date)
                    plt.xlabel("lon [$^\circ$]")
                    plt.ylabel("lat [$^\circ$]")

                    plt.savefig(
                        os.path.join(
                            nc_path,
                            fname + name.replace(".nc", ""),
                            fname + name.replace(".nc", "") + "{}.png".format(ii),
                        )
                    )
                    plt.close()
        logging.info("Finished")
    return


def get_wind(
    start,
    end,
    N_t,
    wind_loopuptable,
    nc_path,
    fname,
    download=True,
    wind_source="knmi_harmonie40",
    username="deltares",
    password="...",
    fig=True,
    hindcast=0,
):
    """download data from matroos and generate input windfield netCDF

    Args:
        start (datetime): starttime
        end (datetime): endtime
        N_t (int): number of slices which are used from each netCDF
        wind_loopuptable (array): array with looputable for pseudo wind
        nc_path (string): path of input netCDF
        fname (string): name of file
        download (bool, optional): download data. Defaults to True.
        wind_source (string, optional): matroos source file. Defaults to 'knmi_harmonie40'.
        username (str, optional): matroos username. Defaults to 'deltares'.
        password (str, optional): matroos password. Defaults to '...'.
        fig (bool, optional): generate figures Defaults to True.
    """

    nc_dates = get_anal_time(start, end, databse="maps2d", source=wind_source)

    assert len(nc_dates) > 0, "No data available"

    if download:
        logging.info("download wind data")
        download_matroos(
            "data_wind",
            nc_dates,
            source=wind_source,
            username=username,
            password=password,
            database="maps2d",
            hindcast=hindcast,
        )

    dates = np.arange(start, end, timedelta(hours=3)).astype(np.datetime64)

    ## create netCDF
    nc_wind = Dataset(
        os.path.join(nc_path, fname + "_wind.nc"), "w", format="NETCDF3_CLASSIC"
    )

    nc_wind.Conventions = "CF-1.6,UGRID-0.9"
    nc_wind.Metadata_Conventions = "Unidata Dataset Discovery v1.0"

    nc_wind.createDimension("time", len(dates))
    nc_wind.createDimension("analysis_time", 1)
    nc_wind.createDimension("y", 753)
    nc_wind.createDimension("x", 672)
    ## variables, time
    time = nc_wind.createVariable("time", np.float64, ("time",))
    time.units = "minutes since 1970-01-01 00:00:00.0 +0000"
    time.long_name = "time"
    time.standard_name = "time"
    time.axis = "T"

    for ii, item_t in enumerate(dates[:]):
        time[ii] = date2num(
            item_t.item(), units="minutes since 1970-01-01 00:00:00.0 +0000"
        )
    ## variables, analysis_time
    analysis_time = nc_wind.createVariable(
        "analysis_time", np.float64, ("analysis_time",)
    )
    analysis_time.units = "1970-01-01 00:00:00.0 +0000"
    analysis_time.long_name = "forecast_reference_time"
    analysis_time.standard_name = "forecast_reference_time"
    ## variables, y
    y = nc_wind.createVariable("y", np.float64, ("y",), fill_value=9.96921e36)
    y.units = "degrees_north"
    y.long_name = "y coordinate according to WGS 1984"
    y.standard_name = "latitude"
    y.axis = "Y"
    ## variables, x
    x = nc_wind.createVariable("x", np.float64, ("x",), fill_value=9.96921e36)
    x.units = "degrees_east"
    x.long_name = "x coordinate according to WGS 1984"
    x.standard_name = "longitude"
    x.axis = "X"
    ## variables, z
    z = nc_wind.createVariable("z", np.float64, ("y", "x"), fill_value=9.96921e36)
    z.units = "meters"
    z.long_name = "height above mean sea level"
    ## variables, crs
    crs = nc_wind.createVariable("crs", np.int32)
    crs.units = "degrees_north"
    crs.long_name = "coordinate reference system"
    crs.grid_mapping_name = "latitude_longitude"
    crs.longitude_of_prime_meridian = 0.0
    crs.semi_major_axis = 6378137.0
    crs.inverse_flattening = 298.257223563
    crs.epsg_code = "EPSG:4326"
    crs.proj4_params = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    crs.crs_wkt = """GEOGCS["WGS 84",
        DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]] """
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
    logging.info("Starting writing input netCDF")
    for jj, item in enumerate(nc_dates):
        nc = Dataset(os.path.join("data_wind", "knmi_harmonie40_" + item + ".nc"))
        ## get dates
        time_dummy = nc.variables["time"]
        date_dummy = np.empty(len(time_dummy[:]), dtype="datetime64[m]")
        for ii, item_t in enumerate(time_dummy[:]):
            dummy = num2date(item_t, units=time_dummy.units)
            date_dummy[ii] = date2num(dummy, units=time.units)
        ## write coordinates
        if jj == 0:
            x = nc.variables["x"][:]
            y = nc.variables["y"][:]
            nc_wind.variables["y"][:] = y
            nc_wind.variables["x"][:] = x

        for ii in range(N_t):
            index = np.where(dates == date_dummy[ii])[0]

            if len(index) == 0:
                continue

            windstress_u = nc.variables["windstress_u"][ii, :, :]  # N/m2
            windstress_v = nc.variables["windstress_v"][ii, :, :]

            u, v = create_pseudowind(
                windstress_u, windstress_v, os.path.join("pseudowind", wind_loopuptable)
            )

            nc_wind.variables["eastward_wind"][index, :, :] = u.data
            nc_wind.variables["northward_wind"][index, :, :] = v.data

        nc.close()
    nc_wind.close()
    logging.info("generate figures")
    if fig:
        ##
        nc_wind = Dataset(os.path.join(nc_path, fname + "_wind.nc"), "r")
        X = nc_wind.variables["x"][:]
        Y = nc_wind.variables["y"][:]
        times = nc_wind.variables["time"]

        dates = np.empty(len(times[:]), dtype="datetime64[m]")
        for ii, item_t in enumerate(times[:]):
            dates[ii] = num2date(int(item_t), units=time.units)

        for ii, date in enumerate(dates):
            ux = nc_wind.variables["eastward_wind"][ii, :, :]
            uy = nc_wind.variables["northward_wind"][ii, :, :]

            u_abs = np.sqrt(ux**2 + uy**2)
            plt.figure(figsize=[10, 10])
            plt.pcolor(X, Y, u_abs)
            plt.colorbar()
            plt.quiver(X[::20], Y[::20], ux[::20, ::20], uy[::20, ::20])
            plt.axis("equal")
            plt.title(date)
            plt.xlabel("lon [$^\circ$]")
            plt.ylabel("lat [$^\circ$]")

            if not os.path.exists(os.path.join(nc_path, fname)):
                os.mkdir(os.path.join(nc_path, fname))

            plt.savefig(
                os.path.join(nc_path, fname.replace(".nc", ""), "{}.png".format(ii))
            )
            plt.close()
        nc_wind.close()
    logging.info("Finished")


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


def get_wind_transformCoord(
    start,
    end,
    wind_lookuptable,
    nc_path,
    fname,
    download=True,
    wind_source="knmi_harmonie40",
    username="deltares",
    password="...",
    fig=True,
    hindcast=0,
    transform=None,
    interpolate_to_grid=None,
    mask_lon=[1.5, 5.5],
    mask_lat=[50.5, 54.5],
):
    """
    download data from matroos and generate input windfield netCDF
    Args:
        start (datetime): starttime
        end (datetime): endtime
        wind_lookuptable (array): array with looputable for pseudo wind
        nc_path (string): path of input netCDF
        fname (string): name of file
        download (bool, optional): download data. Defaults to True.
        wind_source (string, optional): matroos source file. Defaults to 'knmi_harmonie40'.
        username (str, optional): matroos username. Defaults to 'deltares'.
        password (str, optional): matroos password. Defaults to '...'.
        fig (bool, optional): generate figures Defaults to True.
        transform (pyproj.Transformer): for the coordinate transformation
        interpolate_to_grid ([np.array, np.array]): [xgrid,ygrid] coordinates of the grid to interpolate to
        mask_lon ([float, float]): [min_lon, max_lon] for masking to select area of interest
        mask_lat ([float, float]): [min_lat, max_lat] for masking to select area of interest

    Compared to base function 'get_wind', this function also:
        - converts WGS84 coordinates to user defined coordinates
        - uses all timeslices (N_t) in a single netcdf to create the final netcdf
        - defines N_t when calling the function, no longer user-specified

    Based on 'get_wind' from swantools.py from the SWANtoolbox

    Future plans:
        - Shift to xarray (now using nc4)
        - try xr.open_mfdataset() or use current process (making a new nc) or use xr.merge
        - Add user specified bounds of lat/lon to use for slicing

    """
    ## define the projection transformer
    transformer = transform

    ## define the grid to interpolate to
    if interpolate_to_grid is None:
        assert "function must be called with interpolate_to_grid defined"
    else:
        xgrid, ygrid = interpolate_to_grid

    ## get available nc from matroos
    nc_dates = get_anal_time(start, end, databse="maps2d", source=wind_source)
    assert len(nc_dates) > 0, "No data available"
    if download:
        logging.info("download wind data")
        download_matroos(
            "data_wind",
            nc_dates,
            source=wind_source,
            username=username,
            password=password,
            database="maps2d",
            hindcast=hindcast,
        )

    # Determine unique dates
    nc_dates_exists = []
    indices_per_netcdf = []
    for jj, date in enumerate(nc_dates):
        nc = Dataset(os.path.join("data_wind", wind_source + "_" + date + ".nc"), "r")
        date = datetime.strptime(date, "%Y%m%d%H%M")

        time_nc = nc.variables["time"]
        dates_nc = np.empty(len(time_nc[:]), dtype="datetime64[m]")
        for ii, item_t in enumerate(time_nc[:]):
            dates_nc[ii] = num2date(int(item_t), units=time_nc.units)
        if jj + 1 < len(nc_dates):
            next_date = datetime.strptime(nc_dates[jj + 1], "%Y%m%d%H%M")
            index = np.where(dates_nc < next_date)[0]
        else:
            index = np.where(dates_nc < end)[0]

        nc.close()
        if len(index) == 0:
            index = -1
        indices_per_netcdf.append(index)
        ## append dates
        for item in dates_nc[index]:
            nc_dates_exists.append(item)

    dates = np.asarray(nc_dates_exists)
    dates = np.unique(dates)

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
    crs.crs_wkt = f"{transformer.target_crs}"
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

    # ## Define time
    # for ii, item_t in enumerate(dates[:]):
    #     time[ii] = date2num(
    #         item_t.item(), units="minutes since 1970-01-01 00:00:00.0 +0000"
    #     )

    ## write data
    logging.info("Start writing input netCDF")
    # temp_time_index = 0
    for jj, item in enumerate(nc_dates):
        # read individual nc file
        nc = Dataset(os.path.join("data_wind", wind_source + "_" + item + ".nc"))

        ## get dates
        time_dummy = nc.variables["time"]
        date_dummy = np.empty(len(time_dummy[:]), dtype="datetime64[m]")
        for ii, item_t in enumerate(time_dummy[:]):
            dummy = num2date(int(item_t), units=time_dummy.units)
            date_dummy[ii] = date2num(dummy, units=time.units)

        ## write coordinates
        if jj == 0:
            # SWAN does not use lat/lon in this case
            logging.info("lat/lon are left empty")

            lon_temp = nc.variables["x"][:].flatten()
            lat_temp = nc.variables["y"][:].flatten()

            mask = (
                (lon_temp > mask_lon[0])
                & (lon_temp < mask_lon[1])
                & (lat_temp > mask_lat[0])
                & (lat_temp < mask_lat[1])
            )

            lon_temp = lon_temp[mask]
            lat_temp = lat_temp[mask]

            # conversion lat/lon to RD new for y/x
            x_temp, y_temp = transformer.transform(lon_temp, lat_temp)

            nc_wind.variables["y"][:] = ygrid[:, 0]
            nc_wind.variables["x"][:] = xgrid[0, :]

        ## get dates and windstress
        if "windstress_u" in nc.variables and "windstress_v" in nc.variables:
            for ii in range(len(indices_per_netcdf[jj])):
                index = np.where(dates == date_dummy[indices_per_netcdf[jj][ii]])[0][0]

                # Define time
                time[index] = date2num(
                    date_dummy[indices_per_netcdf[jj][ii]].item(),
                    units="minutes since 1970-01-01 00:00:00.0 +0000",
                )

                try:
                    windstress_u = nc.variables["windstress_u"][
                        indices_per_netcdf[jj][ii], :, :
                    ].flatten()
                    windstress_v = nc.variables["windstress_v"][
                        indices_per_netcdf[jj][ii], :, :
                    ].flatten()

                    u, v = create_pseudowind(
                        windstress_u[mask],
                        windstress_v[mask],
                        os.path.join("pseudowind", wind_lookuptable),
                    )

                    u_interp = interpolate.griddata(
                        (x_temp, y_temp),
                        u.data.flatten(),
                        (xgrid, ygrid),
                        method="linear",
                    )
                    v_interp = interpolate.griddata(
                        (x_temp, y_temp),
                        v.data.flatten(),
                        (xgrid, ygrid),
                        method="linear",
                    )

                    nc_wind.variables["eastward_wind"][index, :, :] = u_interp
                    nc_wind.variables["northward_wind"][index, :, :] = v_interp
                except Exception as e:
                    print(f"Error creating pseudowind at index {ii}: {e}")
        else:
            print("windstress_u or windstress_v not found. Using fallback wind data.")
            for ii in range(len(indices_per_netcdf[jj])):
                index = np.where(dates == date_dummy[indices_per_netcdf[jj][ii]])[0][0]

                # Define time
                time[index] = date2num(
                    date_dummy[indices_per_netcdf[jj][ii]].item(),
                    units="minutes since 1970-01-01 00:00:00.0 +0000",
                )

                ## interpolate data from x2d,y2d to xgrid and ygrid
                eastwind_interp = interpolate.griddata(
                    (x_temp, y_temp),
                    nc.variables["eastward_wind"][
                        indices_per_netcdf[jj][ii], :, :
                    ].flatten()[mask],
                    (xgrid, ygrid),
                    method="linear",
                )
                northwind_interp = interpolate.griddata(
                    (x_temp, y_temp),
                    nc.variables["northward_wind"][
                        indices_per_netcdf[jj][ii], :, :
                    ].flatten()[mask],
                    (xgrid, ygrid),
                    method="linear",
                )

                nc_wind.variables["eastward_wind"][index, :, :] = eastwind_interp
                nc_wind.variables["northward_wind"][index, :, :] = northwind_interp

        nc.close()

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
    nc_wind.close()


def get_bc(
    start,
    end,
    N_t,
    source,
    bndloc,
    path_bnd,
    download=True,
    username="deltares",
    password="...",
    nc_x="x",
    nc_y="y",
    nc_e="sea_surface_wave_spectra",
    nc_d="sea_surface_wave_spectral_direction",
    nc_f="sea_surface_wave_spectral_frequency",
    hindcast=0,
):
    """download data and generate boundary conditions

    Args:
        start (datetime): starttime
        end (datetime): endtime
        N_t (int): number of slices which are used from each netCDF
        source (string): matroos source file
        bndloc (array): 2D arrayu with the coordinates of the boundary points
        path_bnd (string): location of boundary file
        download (bool, optional): download data. Defaults to True.
        username (str, optional): matroos username. Defaults to 'deltares'.
        password (str, optional): matroos password. Defaults to '...'.
        nc_x (str, optional): variable in netCDf with the x-coordinates. Defaults to 'x'.
        nc_y (str, optional): variable in netCDf with the y-coordinates. Defaults to 'y'.
        nc_e (str, optional): variable in netCDf with the energy desnity spectrum. Defaults to 'sea_surface_wave_spectra'.
        nc_d (str, optional): variable in netCDf with the direction. Defaults to 'sea_surface_wave_spectral_direction'.
        nc_f (str, optional): variable in netCDf with the frequency. Defaults to 'sea_surface_wave_spectral_frequency'.
    """

    ## get netCDF
    nc_dates = get_anal_time(start, end, databse="maps2d", source=source)

    assert len(nc_dates) > 0, "No data available"

    if download:
        ## download netCDF
        download_matroos(
            "data_bc",
            nc_dates,
            source=source,
            username=username,
            password=password,
            database="maps2d",
            hindcast=hindcast,
        )

    ## number of files
    N_file = len(nc_dates)
    N_loc = len(bndloc)
    ## dimensions: time, dir, freq, (y, x)
    E = np.ones((N_t * N_file, 36, 36, N_loc)) * np.nan

    date_all = np.arange(start, end, timedelta(hours=3)).astype(np.datetime64)
    ## array with dates
    # date_all = np.array([],dtype='datetime64[m]')
    for jj, item in enumerate(nc_dates):
        nc = Dataset(os.path.join("data_bc", source + "_" + item + ".nc"))
        time = nc.variables["time"]

        ## get data from N_t slices
        date = np.empty(len(time[:]), dtype="datetime64[m]")
        for ii, item_t in enumerate(time[:]):
            date[ii] = num2date(int(item_t), units=time.units)

        # date_all = np.append(date_all, date[0:N_t]) # does not work when N_t changes!!

        x = nc.variables[nc_x][:]
        y = nc.variables[nc_y][:]
        X, Y = np.meshgrid(x, y)
        # adjust N_t if times in netCDF are not consistent
        if nc.variables[nc_e].shape[0] < N_t:
            logging.warning(
                "{} does not have {} number of time elements".format(item, N_t)
            )
            N_t0 = nc.variables[nc_e].shape[0]
        else:
            N_t0 = N_t

        for ii, loc in enumerate(bndloc):
            distance = np.sqrt((X - bndloc[ii, 0]) ** 2 + (Y - bndloc[ii, 1]) ** 2)

            index = np.unravel_index(np.argmin(distance, axis=None), distance.shape)

            E_dummy = nc.variables[nc_e][0:N_t, :, :, index[0], index[1]]

            E_dummy[E_dummy == np.inf] = 0

            E[jj * N_t0 : (jj + 1) * N_t0, :, :, ii] = E_dummy

    D = nc.variables[nc_d][:]
    D[D > 180] = D[D > 180] - 360
    freq = nc.variables[nc_f][:]
    nc.close()
    # E[E==np.inf] = -99
    E[np.isnan(E)] = -99

    write_sp2(path_bnd, date_all, E, bndloc[:, 0], bndloc[:, 1], freq, D)

    E_sum = np.apply_over_axes(np.sum, E, [1, 2])
    plt.figure(figsize=[10, 10])
    plt.pcolor(np.linspace(1, len(bndloc), len(bndloc)), date_all, np.squeeze(E_sum))
    plt.colorbar()
    plt.xlabel("bnd loc number")
    plt.ylabel("time axis")
    plt.title("sum E")
    plt.savefig(path_bnd.replace(".bnd", ".png"))


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
