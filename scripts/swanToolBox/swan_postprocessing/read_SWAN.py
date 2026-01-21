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
        names = next((line.strip("%").strip() for i, line in enumerate(file) if i == 4), None)

    # Separate the individual column names to create a list
    tab_columns = names.split()

    # Read .tab file into pandas DataFrame
    dateparse = lambda x: (datetime.datetime.strptime(x, "%Y%m%d.%H%M%S") if "Time" in tab_columns else None)
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
        tab_data = tab_data[~tab_data.index.duplicated(keep='first')]

    # Convert pandas DataFrame to xarray DataArray
    tab_data = tab_data.to_xarray()

    return tab_data


