import pandas as pd
import datetime
import os
import numpy as np
from netCDF4 import Dataset
import xarray as xr
import datetime as dt
import re
from datetime import timedelta
import datetime


def read_SWAN_tab(tab_path: str) -> xr.DataArray:
    """
    Read SWAN .tab files and convert them into an xarray DataArray.

    Parameters:
        tab_path (str): Path to the SWAN .tab file.

    Returns:
        xr.DataArray: DataArray containing the data from the .tab file.
    """

    # Read the tab file and select the fifth row as the column names, skip the %
    with open(tab_path, "r") as file:
        names = next(
            (line.strip("%").strip() for i, line in enumerate(file) if i == 4), None
        )

    # Separate the individual column names to create a list
    tab_columns = names.split()

    # Read .tab file into pandas DataFrame
    dateparse = lambda x: (
        datetime.datetime.strptime(x, "%Y%m%d.%H%M%S")
        if "Time" in tab_columns
        else None
    )
    tab_data = pd.read_csv(
        tab_path,
        skiprows=7,
        header=5,
        sep="\s+",
        names=tab_columns,
        index_col=False,
        parse_dates=["Time"] if "Time" in tab_columns else False,
        date_parser=dateparse,
    )

    x_column = get_column_name("x", tab_data)
    y_column = get_column_name("y", tab_data)
    time_column = get_column_name("time", tab_data)

    # Convert time column to datetime if present
    if time_column:
        # Set index as two spatial dimensions and one temporal dimension
        tab_data.set_index([x_column, y_column, time_column], inplace=True)
    else:
        # Set index as two spatial dimensions
        tab_data.set_index([x_column, y_column], inplace=True)

    # Check for duplicates and handle them
    if tab_data.index.duplicated().any():
        print("Duplicates found in index. Keeping the first occurrence.")
        tab_data = tab_data[~tab_data.index.duplicated(keep="first")]

    # Convert pandas DataFrame to xarray DataArray
    tab_data = tab_data.to_xarray()

    return tab_data


def get_column_name(column_type, data):
    """
    Helper function that obtains the correct column name by checking for several common column names for each parameter.
    """

    # Dictionary with common column names per variable, case does not matter
    column_map = {
        "x": ["X", "Xp", "longitude", "lon"],
        "y": ["Y", "Yp", "latitude", "lat"],
        "time": ["time"],
        "Hs": ["Hm0", "Hm0_observed", "Hs", "Hsig", "wave_height"],
        "Tmm10": ["Tmm10", "Tm10_observed", "Tm_10"],
        "He10": ["He10", "HE10_observed", "Hswell", "hswe"],
        "Dir": ["dir", "meandir", "theta", "theta0"],
        "Dirspr": ["dirspread", "dirspr", "Dspr", "spread", "spr"],
        "Tm01": ["tm01"],
        "Tm02": ["tm02"],
    }

    # Names for the specific parameter
    possible_names = column_map.get(column_type, [])

    # Convert to lower case for case-insensitive matching
    if isinstance(data, pd.DataFrame):
        data_keys = data.columns
        data_dims = []
        data_coords = []
    elif isinstance(data, xr.DataArray) or isinstance(data, xr.Dataset):
        data_keys = data.data_vars.keys()
        data_dims = data.dims
        data_coords = data.coords
    else:
        raise ValueError(
            "Unsupported data type. Expected a pandas DataFrame or xarray DataArray."
        )

    lower_case_keys = {key.lower(): key for key in data_keys}
    lower_case_dims = {dim.lower(): dim for dim in data_dims}
    lower_case_coords = {coord.lower(): coord for coord in data_coords}

    for name in possible_names:
        lower_name = name.lower()
        if lower_name in lower_case_keys:
            return lower_case_keys[lower_name]
        if lower_name in lower_case_dims:
            return lower_case_dims[lower_name]
        if lower_name in lower_case_coords:
            return lower_case_coords[lower_name]

    return None
