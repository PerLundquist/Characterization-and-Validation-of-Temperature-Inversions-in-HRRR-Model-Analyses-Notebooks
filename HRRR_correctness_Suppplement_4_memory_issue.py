#imports
from herbie import Herbie                               # main Herbie Module
from toolbox.gridded_data import pluck_points           # finds the nearest lat/lon point. Comes from Brian Blaylock's Carpenter Workshop project
from synoptic.services import stations_metadata         # Connects to Synoptic API. Comes from Blaylock's SynopticPy project

import os                                               # A library for managing files and directories. Comes from the Standard Python Libraries
import csv                                              # Self explanatory. A library for handling csv files

import math                                                  # A math library from the Standar Python Libraries
import numpy as np                                           # Numerical Python is a common number handling tool used in python
import pandas as pd                                          # A popular data analytics tool in python developed by Wes McKinney. Here it is used to handle array objects such as upper air soundings
pd.set_option('display.min_rows', 30)                        # Changing the default number of rows in a dataframe that pandas will display

from datetime import datetime                                # A module from the standard python library for handling date objects. Will be used to import data at specific times, but is also handy in iterating through dates
from datetime import timedelta                               # A datetime module needed to advance a datetime object
from siphon.simplewebservice.wyoming import WyomingUpperAir  # Siphon is a tool developed by Unidata for querying data from various THREDDS servers. This particular module accesses data from the THREDDS server owned by the University of Wyoming



# station identifier names formatted such that the Herbie or Siphon libraries can properly read them
stids_HRRR    = ['KUNR', 'KRIW', 'KGGW', 'KDNR']
stids_OBS     = ['RAP',  'RIW',  'GGW',  'DNR']

# Time domain
start_time    = datetime.strptime('2017-01-01_12', "%Y-%m-%d_%H")
stop_time     = datetime.strptime('2023-01-01_12', "%Y-%m-%d_%H")

# local directory holding sounding data
sounding_dir  = './sounding_csvs/'


#########################################
### GRABBING THE HRRR DATA VIA HERBIE ###
#########################################

# A revised key_err_fix() function that should grab the index for the xarray dataframe we want from Herbie, regardless
# of the coordinate system
def key_err_fix(dataframe, key_, orog_check=0):
    for i in range(len(dataframe)):

        coords_keys = list(dataframe[i].coords.keys())
        data_vars_keys = list(dataframe[i].data_vars.keys())

        if orog_check == 1:
            if any(ele in data_vars_keys for ele in [key_]):
                return i
        else:
            if any(ele in coords_keys for ele in ['isobaricInhPa']) and any(ele in data_vars_keys for ele in [key_]):
                return i


# A function for grabbing Herbie data depending on the variable we want
# run_time = the date of interest
# var      = the requested data variable, in this case temperature or orography
# stid     = the Herbie compatible station identifier. Typically, it's different than the one used in siphon
# model_   = the NWP model to be used
# product_ = the type of product coordinates to be used
def get_HRRR_data(run_time, var, stid, model_='HRRR', product_='prs'):
    print('\nGrabbing HRRR ' + var + ' data for ' + run_time)

    # A lookup table with the appropriate data variable key from the xarray dataframe needed
    match var:
        case "TMP":
            key_ = 't'
        case "DPT":
            key_ = 'dpt'
        case "U":
            key_ = 'u'
        case "V":
            key_ = 'v'
        case "W":
            var = "V"
            key_ = 'w'
        case "RH":
            key_ = 'r'
        case "HGT":
            key_ = 'gh'
        case "OROG":
            var = "HGT"
            key_ = 'orog'

    # Get HRRR dataframe
    try:
        # make initial Herbie object
        H_xf = Herbie(run_time, model=model_, product=product_, verbose=False).xarray(var)

        # in order to use key_err_fix, the Herbie object must be a list of xarray dataframes. If there's only one
        # dataframe, Herbie won't put it in a list, so we'll fix that here
        if isinstance(H_xf, list):
            pass
        else:
            H_xf = [H_xf]

    except Exception as e:
        print('time ' + run_time + ' could not be collected. Omitting from records............\n')

        if key_ == 'orog':
            return 'nan'
        else:
            return 'nan', 'nan', 'nan'

    # set lookup index
    if key_ == 'orog':
        index_ = key_err_fix(H_xf, key_, orog_check=1)
    else:
        index_ = key_err_fix(H_xf, key_)

    try:

        # Get Point Data
        closest_staions_array = stations_metadata(radius=stid + ",0", verbose=False)
        station_points = np.array(
            list(zip(closest_staions_array.loc["longitude"], closest_staions_array.loc["latitude"])))
        station_names = closest_staions_array.loc["STID"].to_numpy()

        # make secondary xarray object with vertical profile of the variable we want a the location we want
        ds_pluck = pluck_points(H_xf[index_], station_points[:1], station_names[:1],
                                verbose=False)  # see lookup table in above cell

        # confirm we have the coordinate system we want
        if any(coord_name == 'isobaricInhPa' for coord_name in ds_pluck.coords):

            # number of levels to work with
            levels = len(ds_pluck.coords['isobaricInhPa'])

            # save data to lists. IT'S IMPORTANT THAT THEY'RE LISTS NOT NUMPY ARRAYS YET!!!
            var_out = []
            pres = []

            for i in range(levels):
                var_out.append(float(ds_pluck.data_vars[key_][0, i]))  # see lookup table above
                pres.append(float(ds_pluck.coords['isobaricInhPa'][i]))

            # return the variables
            return var_out, pres, levels

        # just return one variable if different coords
        else:

            return float(ds_pluck.data_vars[key_])


    except Exception as e:
        print('time ' + run_time + ' could not be collected. Omitting from records............\n')

        if key_ == 'orog':
            return 'nan'
        else:
            return 'nan', 'nan', 'nan'


# Grab a raw modelled profile
def mk_HRRR_df(curr_date, stid, directory):
    print('\nGrabbing HRRR data for ' + curr_date)

    # Check if a file for the date exists
    local_available = False

    input_date = datetime.strptime(curr_date, "%Y-%m-%d %H:%M")
    file_name = input_date.strftime("%Y-%m-%d_%HUTC.csv")

    file_loc = directory + stid + '/HRRR/'

    file_path = os.path.join(file_loc, file_name)

    if os.path.isfile(file_path):
        local_available = True

    # Download the profile if no local file available
    if not local_available:
        #         # get HRRR data
        #         height_HRRR, pres_HRRR, levels_HRRR = get_HRRR_data(curr_date, 'HGT', stid)
        #         temp_HRRR, pres_HRRR, levels_HRRR = get_HRRR_data(curr_date, 'TMP', stid)
        #         dwpt_HRRR, pres_HRRR, levels_HRRR = get_HRRR_data(curr_date, 'DPT', stid)
        #         u_HRRR, pres_HRRR, levels_HRRR = get_HRRR_data(curr_date, 'U', stid)
        #         v_HRRR, pres_HRRR, levels_HRRR = get_HRRR_data(curr_date, 'V', stid)

        #         # surface elev (labeled Geopotential Height - Orography in the file)
        #         sfc_elev = get_HRRR_data(curr_date, 'OROG', stid)

        #         # Check if data collection was successful:
        #         list_check = [sfc_elev, height_HRRR, temp_HRRR, dwpt_HRRR, u_HRRR, v_HRRR, pres_HRRR, levels_HRRR, sfc_elev]
        #         bool_check = False

        #         for ele in list_check:
        #             if type(ele) == str:
        #                 if ele == 'nan':
        #                     bool_check = True
        #                     break

        #         if bool_check:
        #             return 'nan'

        #         # adjust temps because they're in K
        #         # adjust height because it's in units of meters above msl
        #         temp_HRRR = [temp - 273.15 for temp in temp_HRRR]
        #         dwpt_HRRR = [dwpt - 273.15 for dwpt in dwpt_HRRR]

        #         height_HRRR = [height - sfc_elev for height in height_HRRR]

        #         # put it all in a dataframe
        #         HRRR_df = pd.DataFrame({
        #             'Pressure level (hPa)': pres_HRRR,
        #             'Height AGL (m)': height_HRRR,
        #             'Temp (°C)': temp_HRRR,
        #             'Dwpt (°C)': dwpt_HRRR,
        #             'U wind (m/s)': u_HRRR,
        #             'V wind (m/s)': v_HRRR
        #         })

        #         # save the dataframe to file
        #         if not os.path.exists(directory):
        #             os.mkdir(directory)
        #         if not os.path.exists(directory + stid):
        #             os.mkdir(directory + stid)
        #         if not os.path.exists(file_loc):
        #             os.mkdir(file_loc)

        #         HRRR_df.to_csv(file_path, index=False)

        #         return HRRR_df

        print("no HRRR data available")
        return 'nan'


    # Grab the file if it's found
    else:
        HRRR_df = pd.read_csv(file_path, encoding='latin-1')

        # Sometimes the temp and dwpt columns are labelled poorly. This will fix that
        if 'Temp (°C)' not in HRRR_df.columns:
            alt_label = 'Temp (Â°C)'
            if alt_label in HRRR_df.columns:
                HRRR_df.rename(columns={alt_label: 'Temp (°C)'}, inplace=True)

        if 'Dwpt (°C)' not in HRRR_df.columns:
            alt_label = 'Dwpt (Â°C)'
            if alt_label in HRRR_df.columns:
                HRRR_df.rename(columns={alt_label: 'Dwpt (°C)'}, inplace=True)

        # Use the first height in this entry to adjust to height AGL
        sfc_elev = float(HRRR_df['Height AGL (m)'].iloc[0])
        HRRR_df['Height AGL (m)'] = [h - sfc_elev for h in HRRR_df['Height AGL (m)']]

        return HRRR_df


#############################################
### GRABBING THE OBSERVED DATA VIA SIPHON ###
#############################################
def get_OBS_data(run_time, stid, obs_stid, directory):
    print('\nGrabbing OBS data for ' + run_time)

    # Check if a file for the date exists
    local_available = False

    input_date = datetime.strptime(run_time, "%Y-%m-%d %H:%M")
    file_name = input_date.strftime("%Y-%m-%d_%HUTC.csv")

    file_loc = directory + stid + '/OBS/'

    file_path = os.path.join(file_loc, file_name)

    if os.path.isfile(file_path):
        local_available = True

    try:

        if not local_available:
            # Provide date/time: year, month, day, hour
            date = datetime.strptime(run_time, "%Y-%m-%d %H:%M")

            # And download the data
            df = WyomingUpperAir.request_data(date, obs_stid)

            # adjust to be in meters AGL
            sfc_elev = df['height'].iloc[0]
            df['height'] = [h - sfc_elev for h in df['height']]

            # Define some variables to make things easy
            p = df['pressure'].values
            h = df['height'].values
            t = df['temperature'].values
            td = df['dewpoint'].values
            u = df['u_wind'].values
            v = df['v_wind'].values

            better_df = pd.DataFrame({
                'Pressure level (hPa)': p,
                'Height AGL (m)': h,
                'Temp (°C)': t,
                'Dwpt (°C)': td,
                'U wind (m/s)': u,
                'V wind (m/s)': v
            })

            # save the dataframe to file
            if not os.path.exists(directory):
                os.mkdir(directory)
            if not os.path.exists(directory + stid):
                os.mkdir(directory + stid)
            if not os.path.exists(file_loc):
                os.mkdir(file_loc)

            better_df.to_csv(file_path, index=False)

            return better_df

        # Grab the file if it's found
        else:
            OBS_df = pd.read_csv(file_path, encoding='latin-1')

            # Sometimes the temp and dwpt columns are labelled poorly. This will fix that
            if 'Temp (°C)' not in OBS_df.columns:
                alt_label = 'Temp (Â°C)'
                if alt_label in OBS_df.columns:
                    OBS_df.rename(columns={alt_label: 'Temp (°C)'}, inplace=True)

            if 'Dwpt (°C)' not in OBS_df.columns:
                alt_label = 'Dwpt (Â°C)'
                if alt_label in OBS_df.columns:
                    OBS_df.rename(columns={alt_label: 'Dwpt (°C)'}, inplace=True)

            sfc_elev = OBS_df['Height AGL (m)'].iloc[0]
            OBS_df['Height AGL (m)'] = [h - sfc_elev for h in OBS_df['Height AGL (m)']]

            return OBS_df

    except Exception as e:
        return 'nan'


########################################################
### SLICE THE DATAFRAMES TO START AT THE SAME HEIGHT ###
########################################################
def start_at_same_height(OBS_df, HRRR_df):
    new_OBS_df = OBS_df
    new_HRRR_df = HRRR_df

    first_obs_height = new_OBS_df['Height AGL (m)'].iloc[0]
    first_hrrr_height = new_HRRR_df['Height AGL (m)'].iloc[0]

    # if the sfc value on the observed data is below the sfc on the hrrr, cut it off
    if first_obs_height < first_hrrr_height:
        obs_sfc_data = pd.DataFrame({
            'Pressure level (hPa)': [
                linear_fix(first_hrrr_height, new_OBS_df['Height AGL (m)'], new_OBS_df['Pressure level (hPa)'],
                           increase_with_height=True)],
            'Height AGL (m)': [first_hrrr_height],
            'Temp (°C)': [
                linear_fix(first_hrrr_height, new_OBS_df['Height AGL (m)'], new_OBS_df['Temp (°C)'],
                           increase_with_height=True)],
            'Dwpt (°C)': [
                linear_fix(first_hrrr_height, new_OBS_df['Height AGL (m)'], new_OBS_df['Dwpt (°C)'],
                           increase_with_height=True)]
        })

        # remove values below the first height in modelled data
        new_OBS_df = new_OBS_df[new_OBS_df['Height AGL (m)'] >= first_hrrr_height]

        # add the new sfc data to it and sort
        new_OBS_df = pd.concat([obs_sfc_data, new_OBS_df])
        new_OBS_df = OBS_df.sort_values(by='Height AGL (m)', ascending=True)
        new_OBS_df = OBS_df.reset_index(drop=True)


    # if the sfc value on the hrrr is below the sfc on the observed data, cut it off
    elif first_obs_height > first_hrrr_height:
        hrrr_sfc_data = pd.DataFrame({
            'Pressure level (hPa)': [
                linear_fix(first_obs_height, new_HRRR_df['Height AGL (m)'], new_HRRR_df['Pressure level (hPa)'],
                           increase_with_height=True)],
            'Height AGL (m)': [first_obs_height],
            'Temp (°C)': [linear_fix(first_obs_height, new_HRRR_df['Height AGL (m)'], new_HRRR_df['Temp (°C)'],
                                     increase_with_height=True)],
            'Dwpt (°C)': [linear_fix(first_obs_height, new_HRRR_df['Height AGL (m)'], new_HRRR_df['Dwpt (°C)'],
                                     increase_with_height=True)]
        })

        # remove values below the first height in observed data
        new_HRRR_df = HRRR_df[new_HRRR_df['Height AGL (m)'] >= first_obs_height]

        # add the new sfc data to it and sort
        new_HRRR_df = pd.concat([hrrr_sfc_data, new_HRRR_df])
        HRRR_df = HRRR_df.sort_values(by='Height AGL (m)', ascending=True)
        HRRR_df = HRRR_df.reset_index(drop=True)

    return new_OBS_df, new_HRRR_df


#########################################################
### THE LINEAR INTERPOLATION FUNCTION USED EVERYWHERE ###
#########################################################

# conduct a linear interpolation between points.

# ref_coord            = the desired location you want to conduct a linear interpolation. (y in the above equation)
# unfixed_data_coords  = array of the coordinate values in the raw data (contains y_above and y_below)
# unfixed_data_values  = array of the variable values from the raw data (contains x_above and x_below)
# increase_with_height = a boolean to tell the function if your height coordinate is increasing or decreasing with
#                        height. Used to switch between height and pressure coordinates
def linear_fix(ref_coord, unfixed_data_coords, unfixed_data_values, increase_with_height=False):
    # Rudimentary error checking. Doesn't stop the program, but won't create an interpolated dataframe
    if len(unfixed_data_coords) != len(unfixed_data_values):
        print('The variable and pressure arrays must be of equal length. Cancelling interpolation......')
        return

    # Height coordinates (function is the same in both scenarios)
    if increase_with_height:

        # scan the data until we find the first point above our reference point
        for i in range(len(unfixed_data_values)):
            if unfixed_data_coords[i] > ref_coord:
                # interpolate temp and rh between pressure levels above/below surface pressure to get surface values
                #
                # Equation: x = x_above + ( ((x_below - x_above) * (height_above - ref_height)) / (height_above - height_below) )

                fixed_value = unfixed_data_values[i] + (((unfixed_data_values[i - 1] - unfixed_data_values[i]) * (
                        unfixed_data_coords[i] - ref_coord)) / (unfixed_data_coords[i] - unfixed_data_coords[
                    i - 1]))
                return fixed_value
                break  # This won't run, but I put it here just in case. Perhaps I have anxiety, haha

    # Pressure coordinates
    else:

        # scan the data until we find the first point above our reference point
        for i in range(len(unfixed_data_values)):
            if unfixed_data_coords[i] < ref_coord:
                # interpolate temp and rh between pressure levels above/below surface pressure to get surface values
                #
                # Equation: x = x_above + ( ((x_below - x_above) * (pressure_above - ref_pressure)) / (pressure_above - pressure_below) )

                fixed_value = unfixed_data_values[i] + (((unfixed_data_values[i - 1] - unfixed_data_values[i]) * (
                        unfixed_data_coords[i] - ref_coord)) / (unfixed_data_coords[i] - unfixed_data_coords[
                    i - 1]))
                return fixed_value
                break  # just in case


#############################################################
### CONVERT THE DATAFRAME TO A STANDARD HEIGHT COORDINATE ###
#############################################################

# Note: instead of locking the cutoff height to 5km, we will allow the option to change the cutoff height. This is used
# later to explore ways of improving the model correctness in creating inversions. Nevertheless, when running the normalization
# function, the domain of [0, 5000] is still used. The cutoff is not utilized there

def mk_height_df(OBS_df, HRRR_df, cutoff=5000.):
    # Make a new dataframe for the OBSERVED data such that datapoints are every 100m
    standard_heights = np.arange(0, cutoff, cutoff / 50.)

    # Make arrays for each variable we care about and conduct linear interpolation
    pressures = []
    temps = []
    dwpts = []

    for h in standard_heights:
        pressures.append(
            linear_fix(h, OBS_df['Height AGL (m)'], OBS_df['Pressure level (hPa)'], increase_with_height=True))
        temps.append(linear_fix(h, OBS_df['Height AGL (m)'], OBS_df['Temp (°C)'], increase_with_height=True))
        dwpts.append(linear_fix(h, OBS_df['Height AGL (m)'], OBS_df['Dwpt (°C)'], increase_with_height=True))

    OBS_df_standard = pd.DataFrame({
        'Pressure level (hPa)': pressures,
        'Height AGL (m)': standard_heights,
        'Temp (°C)': temps,
        'Dwpt (°C)': dwpts
    })

    # set up all the lapse rates through finite differencing

    # initialize a new lapse rate column that is full of zeros
    OBS_df_standard['lapse_rate'] = 0

    # loop through every point in the dataframe, assigning a lapse rate value
    for i in range(len(OBS_df_standard['Temp (°C)'])):

        if i == (len(
                OBS_df_standard['Temp (°C)']) - 1):  # fill the final point. just say it's the dry adiabatic lapse rate
            OBS_df_standard['lapse_rate'].iloc[i] = -9.7670 * (1 / 1000.)  # °C/m
        else:
            OBS_df_standard['lapse_rate'].iloc[i] = (OBS_df_standard['Temp (°C)'].iloc[i + 1] -
                                                     OBS_df_standard['Temp (°C)'].iloc[i]) / (
                                                                OBS_df_standard['Height AGL (m)'].iloc[i + 1] -
                                                                OBS_df_standard['Height AGL (m)'].iloc[i])

    # Do the same to the HRRR data
    pressures = []
    temps = []
    dwpts = []

    for h in standard_heights:
        pressures.append(
            linear_fix(h, HRRR_df['Height AGL (m)'], HRRR_df['Pressure level (hPa)'], increase_with_height=True))
        temps.append(linear_fix(h, HRRR_df['Height AGL (m)'], HRRR_df['Temp (°C)'], increase_with_height=True))
        dwpts.append(linear_fix(h, HRRR_df['Height AGL (m)'], HRRR_df['Dwpt (°C)'], increase_with_height=True))

    HRRR_df_standard = pd.DataFrame({
        'Pressure level (hPa)': pressures,
        'Height AGL (m)': standard_heights,
        'Temp (°C)': temps,
        'Dwpt (°C)': dwpts
    })

    # set up all the lapse rates through finite differencing

    # initialize a new lapse rate column that is full of zeros
    HRRR_df_standard['lapse_rate'] = 0

    # loop through every point in the dataframe, assigning a lapse rate value
    for i in range(len(HRRR_df_standard['Temp (°C)'])):

        if i == (len(
                HRRR_df_standard['Temp (°C)']) - 1):  # fill the final point. just say it's the dry adiabatic lapse rate
            HRRR_df_standard['lapse_rate'].iloc[i] = -9.7670 * (1 / 1000.)  # °C/m
        else:
            HRRR_df_standard['lapse_rate'].iloc[i] = (HRRR_df_standard['Temp (°C)'].iloc[i + 1] -
                                                      HRRR_df_standard['Temp (°C)'].iloc[i]) / (
                                                                 HRRR_df_standard['Height AGL (m)'].iloc[i + 1] -
                                                                 HRRR_df_standard['Height AGL (m)'].iloc[i])

    return OBS_df_standard, HRRR_df_standard


###############################################################
### CONVERT THE DATAFRAME TO A NORMALIZED HEIGHT COORDINATE ###
###############################################################

def mk_normalized_height_df(OBS_df, HRRR_df):
    # take the values in standard_heights and OBS_df_standard and normalize to the specified range
    # Heights will be nomalized from a range of [0, 5000] to [0, 1]
    # Temperatures will be normalized from a range of [-39, 40] to [0, 1]

    normalized_heights = np.arange(0, 1.0, (100 / 5000.))  # (start value, end value, increment)

    temps = np.zeros(shape=len(normalized_heights))
    dwpts = np.zeros(shape=len(normalized_heights))

    for i in range(len(normalized_heights)):

        # adding 39 puts it to a range of [0, 79], then simply divide by 79 to get [0, 1]
        if OBS_df_standard['Temp (°C)'][i] is not None:
            temps[i] = (OBS_df_standard['Temp (°C)'][i] + 39.) / (39. + 40.)
        if OBS_df_standard['Dwpt (°C)'][i] is not None:
            dwpts[i] = (OBS_df_standard['Dwpt (°C)'][i] + 39.) / (39. + 40.)

    OBS_df_normalized = pd.DataFrame({
        'Pressure level (hPa)': OBS_df_standard['Pressure level (hPa)'][i],
        'Height AGL (m)': normalized_heights,
        'Temp (°C)': temps,
        'Dwpt (°C)': dwpts
    })

    # set up all the lapse rates through finite differencing

    # initialize a new lapse rate column that is full of zeros
    OBS_df_normalized['lapse_rate'] = 0

    # loop through every point in the dataframe, assigning a lapse rate value
    for i in range(len(OBS_df_normalized['Temp (°C)'])):

        if i == (len(OBS_df_normalized[
                         'Temp (°C)']) - 1):  # fill the final point. just say it's the dry adiabatic lapse rate
            OBS_df_normalized['lapse_rate'].iloc[i] = -9.7670 * (1 / 1000.)  # °C/m
        else:
            OBS_df_normalized['lapse_rate'].iloc[i] = (OBS_df_normalized['Temp (°C)'].iloc[i + 1] -
                                                       OBS_df_normalized['Temp (°C)'].iloc[i]) / (
                                                                  OBS_df_normalized['Height AGL (m)'].iloc[i + 1] -
                                                                  OBS_df_normalized['Height AGL (m)'].iloc[i])

    # Likewise with HRRR data

    temps = np.zeros(shape=len(normalized_heights))
    dwpts = np.zeros(shape=len(normalized_heights))

    for i in range(len(normalized_heights)):

        # adding 39 puts it to a range of [0, 79], then simply divide by 79 to get [0, 1]
        if HRRR_df_standard['Temp (°C)'][i] is not None:
            temps[i] = (HRRR_df_standard['Temp (°C)'][i] + 39.) / (39. + 40.)
        if HRRR_df_standard['Dwpt (°C)'][i] is not None:
            dwpts[i] = (HRRR_df_standard['Dwpt (°C)'][i] + 39.) / (39. + 40.)

    HRRR_df_normalized = pd.DataFrame({
        'Pressure level (hPa)': HRRR_df_standard['Pressure level (hPa)'][i],
        'Height AGL (m)': normalized_heights,
        'Temp (°C)': temps,
        'Dwpt (°C)': dwpts
    })

    # initialize a new lapse rate column that is full of zeros
    HRRR_df_normalized['lapse_rate'] = 0

    # loop through every point in the dataframe, assigning a lapse rate value
    for i in range(len(HRRR_df_normalized['Temp (°C)'])):

        if i == (len(HRRR_df_normalized[
                         'Temp (°C)']) - 1):  # fill the final point. just say it's the dry adiabatic lapse rate
            HRRR_df_normalized['lapse_rate'].iloc[i] = -9.7670 * (1 / 1000.)  # °C/m
        else:
            HRRR_df_normalized['lapse_rate'].iloc[i] = (HRRR_df_normalized['Temp (°C)'].iloc[i + 1] -
                                                        HRRR_df_normalized['Temp (°C)'].iloc[i]) / (
                                                                   HRRR_df_normalized['Height AGL (m)'].iloc[i + 1] -
                                                                   HRRR_df_normalized['Height AGL (m)'].iloc[i])

    return OBS_df_normalized, HRRR_df_normalized


###############################################
### GRAB THE PRIMARY TEMPERATURE INVERSIONS ###
###############################################
def inversion_grabber(OBS_df_standard, OBS_df_norm, HRRR_df_standard, HRRR_df_norm):
    # Grab the inversion lists in regular coordinates

    # Follow Vector_Method.ipynb to create a plot with both observed and modelled inversion vectors on it
    # We'll just use a list to save the inversions
    OBS_inversions = []

    # booleans to check if we started/finished grabbing an inversion
    start_grabbed = 0
    end_grabbed = 0

    # Scan all points to find positive lapse rates
    for i in range(len(OBS_df_standard['lapse_rate'])):

        # Positive lapse rate found! Save this coordiate as [height, temp]
        if (OBS_df_standard['lapse_rate'][i] >= 0) and (start_grabbed == 0):
            start_coord = [OBS_df_standard['Height AGL (m)'][i], OBS_df_standard['Temp (°C)'][i]]
            start_grabbed = 1  # we just grabbed the start point, so tell the program we did
            end_grabbed = 0  # haven't grabbed the ending point though...

        # The lapse rate above this point is no longer positive.... Guess we hit the end of the inversion
        # Note, this part can only run if we've already started grabbing an inversion
        elif (OBS_df_standard['lapse_rate'][i] < 0) and (start_grabbed == 1):
            end_coord = [OBS_df_standard['Height AGL (m)'][i], OBS_df_standard['Temp (°C)'][i]]
            start_grabbed = 0  # We finished grabbing this inversion, so we can tell the program it's ready to start a new one
            end_grabbed = 1  # Finished grabbing this inversion, so let's say that we grabbed it

        # save the coordinates, ensuring that we finished grabbing this inversion and the program is ready for the next one
        if end_grabbed == 1 and start_grabbed == 0:
            OBS_inversions.append([start_coord, end_coord])
            start_grabbed = 0
            end_grabbed = 0

    # Calculate distances and store them with their respective line segments
    data = []  # simple list to store the data of each inversion. Not organized quite yet

    for line in OBS_inversions:
        point1, point2 = line
        distance = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)  # Distance formula

        # Extract relevant information
        start_height = point1[0]
        start_temp = point1[1]
        end_height = point2[0]
        end_temp = point2[1]

        # Save the data to the list
        data.append([distance, start_height, start_temp, end_height, end_temp])

    # clean up the data by putting it into a pandas dataframe, ranking the inversions, and parsing the inversion of interest
    OBS_inversion_ranks = pd.DataFrame(data,
                                       columns=['Distance', 'Starting Height', 'Starting Temperature', 'Ending Height',
                                                'Ending Temperature'])
    OBS_inversion_ranks = OBS_inversion_ranks.sort_values(by='Distance', ascending=False)
    OBS_inversion_ranks = OBS_inversion_ranks.reset_index(drop=True)

    # And the same for the HRRR
    HRRR_inversions = []

    # booleans to check if we started/finished grabbing an inversion
    start_grabbed = 0
    end_grabbed = 0

    # Scan all points to find positive lapse rates
    for i in range(len(HRRR_df_standard['lapse_rate'])):

        # Positive lapse rate found! Save this coordiate as [height, temp]
        if (HRRR_df_standard['lapse_rate'][i] >= 0) and (start_grabbed == 0):
            start_coord = [HRRR_df_standard['Height AGL (m)'][i], HRRR_df_standard['Temp (°C)'][i]]
            start_grabbed = 1  # we just grabbed the start point, so tell the program we did
            end_grabbed = 0  # haven't grabbed the ending point though...

        # The lapse rate above this point is no longer positive.... Guess we hit the end of the inversion
        # Note, this part can only run if we've already started grabbing an inversion
        elif (HRRR_df_standard['lapse_rate'][i] < 0) and (start_grabbed == 1):
            end_coord = [HRRR_df_standard['Height AGL (m)'][i], HRRR_df_standard['Temp (°C)'][i]]
            start_grabbed = 0  # We finished grabbing this inversion, so we can tell the program it's ready to start a new one
            end_grabbed = 1  # Finished grabbing this inversion, so let's say that we grabbed it

        # save the coordinates, ensuring that we finished grabbing this inversion and the program is ready for the next one
        if end_grabbed == 1 and start_grabbed == 0:
            HRRR_inversions.append([start_coord, end_coord])
            start_grabbed = 0
            end_grabbed = 0

    # Calculate distances and store them with their respective line segments
    data = []  # simple list to store the data of each inversion. Not organized quite yet

    for line in HRRR_inversions:
        point1, point2 = line
        distance = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)  # Distance formula

        # Extract relevant information
        start_height = point1[0]
        start_temp = point1[1]
        end_height = point2[0]
        end_temp = point2[1]

        # Save the data to the list
        data.append([distance, start_height, start_temp, end_height, end_temp])

    # clean up the data by putting it into a pandas dataframe, ranking the inversions, and parsing the inversion of interest
    HRRR_inversion_ranks = pd.DataFrame(data,
                                        columns=['Distance', 'Starting Height', 'Starting Temperature', 'Ending Height',
                                                 'Ending Temperature'])
    HRRR_inversion_ranks = HRRR_inversion_ranks.sort_values(by='Distance', ascending=False)
    HRRR_inversion_ranks = HRRR_inversion_ranks.reset_index(drop=True)

    # Grab the inversion lists in normalized coordinates

    OBS_inversions_norm = []

    # booleans to check if we started/finished grabbing an inversion
    start_grabbed = 0
    end_grabbed = 0

    # Scan all points to find positive lapse rates
    for i in range(len(OBS_df_normalized['lapse_rate'])):

        # Positive lapse rate found! Save this coordiate as [height, temp]
        if (OBS_df_normalized['lapse_rate'][i] >= 0) and (start_grabbed == 0):
            start_coord = [OBS_df_normalized['Height AGL (m)'][i], OBS_df_normalized['Temp (°C)'][i]]
            start_grabbed = 1  # we just grabbed the start point, so tell the program we did
            end_grabbed = 0  # haven't grabbed the ending point though...

        # The lapse rate above this point is no longer positive.... Guess we hit the end of the inversion
        # Note, this part can only run if we've already started grabbing an inversion
        elif (OBS_df_normalized['lapse_rate'][i] < 0) and (start_grabbed == 1):
            end_coord = [OBS_df_normalized['Height AGL (m)'][i], OBS_df_normalized['Temp (°C)'][i]]
            start_grabbed = 0  # We finished grabbing this inversion, so we can tell the program it's ready to start a new one
            end_grabbed = 1  # Finished grabbing this inversion, so let's say that we grabbed it

        # save the coordinates, ensuring that we finished grabbing this inversion and the program is ready for the next one
        if end_grabbed == 1 and start_grabbed == 0:
            OBS_inversions_norm.append([start_coord, end_coord])
            start_grabbed = 0
            end_grabbed = 0

    # Calculate distances and store them with their respective line segments
    data = []  # simple list to store the data of each inversion. Not organized quite yet

    for line in OBS_inversions_norm:
        point1, point2 = line
        distance = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)  # Distance formula

        # Extract relevant information
        start_height = point1[0]
        start_temp = point1[1]
        end_height = point2[0]
        end_temp = point2[1]

        # Save the data to the list
        data.append([distance, start_height, start_temp, end_height, end_temp])

    # clean up the data by putting it into a pandas dataframe, ranking the inversions, and parsing the inversion of interest
    OBS_inversion_ranks_norm = pd.DataFrame(data, columns=['Distance', 'Starting Height', 'Starting Temperature',
                                                           'Ending Height', 'Ending Temperature'])

    OBS_inversion_ranks = OBS_inversion_ranks.sort_values(by='Starting Height', ascending=False)
    OBS_inversion_ranks_norm = OBS_inversion_ranks_norm.sort_values(by='Starting Height', ascending=False)
    OBS_inversion_ranks = OBS_inversion_ranks.reset_index(drop=True)
    OBS_inversion_ranks_norm = OBS_inversion_ranks_norm.reset_index(drop=True)

    OBS_inversion_ranks['Distance'] = OBS_inversion_ranks_norm['Distance']

    OBS_inversion_ranks = OBS_inversion_ranks.sort_values(by='Distance', ascending=False)
    OBS_inversion_ranks = OBS_inversion_ranks.reset_index(drop=True)

    # Likewise with the HRRR data
    HRRR_inversions_norm = []

    # booleans to check if we started/finished grabbing an inversion
    start_grabbed = 0
    end_grabbed = 0

    # Scan all points to find positive lapse rates
    for i in range(len(HRRR_df_normalized['lapse_rate'])):

        # Positive lapse rate found! Save this coordiate as [height, temp]
        if (HRRR_df_normalized['lapse_rate'][i] >= 0) and (start_grabbed == 0):
            start_coord = [HRRR_df_normalized['Height AGL (m)'][i], HRRR_df_normalized['Temp (°C)'][i]]
            start_grabbed = 1  # we just grabbed the start point, so tell the program we did
            end_grabbed = 0  # haven't grabbed the ending point though...

        # The lapse rate above this point is no longer positive.... Guess we hit the end of the inversion
        # Note, this part can only run if we've already started grabbing an inversion
        elif (HRRR_df_normalized['lapse_rate'][i] < 0) and (start_grabbed == 1):
            end_coord = [HRRR_df_normalized['Height AGL (m)'][i], HRRR_df_normalized['Temp (°C)'][i]]
            start_grabbed = 0  # We finished grabbing this inversion, so we can tell the program it's ready to start a new one
            end_grabbed = 1  # Finished grabbing this inversion, so let's say that we grabbed it

        # save the coordinates, ensuring that we finished grabbing this inversion and the program is ready for the next one
        if end_grabbed == 1 and start_grabbed == 0:
            HRRR_inversions_norm.append([start_coord, end_coord])
            start_grabbed = 0
            end_grabbed = 0

    # Likewise with the HRRR data
    # Calculate distances and store them with their respective line segments
    data = []  # simple list to store the data of each inversion. Not organized quite yet

    for line in HRRR_inversions_norm:
        point1, point2 = line
        distance = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)  # Distance formula

        # Extract relevant information
        start_height = point1[0]
        start_temp = point1[1]
        end_height = point2[0]
        end_temp = point2[1]

        # Save the data to the list
        data.append([distance, start_height, start_temp, end_height, end_temp])

    # clean up the data by putting it into a pandas dataframe, ranking the inversions, and parsing the inversion of interest
    HRRR_inversion_ranks_norm = pd.DataFrame(data, columns=['Distance', 'Starting Height', 'Starting Temperature',
                                                            'Ending Height', 'Ending Temperature'])

    HRRR_inversion_ranks = HRRR_inversion_ranks.sort_values(by='Starting Height', ascending=False)
    HRRR_inversion_ranks_norm = HRRR_inversion_ranks_norm.sort_values(by='Starting Height', ascending=False)
    HRRR_inversion_ranks = HRRR_inversion_ranks.reset_index(drop=True)
    HRRR_inversion_ranks_norm = HRRR_inversion_ranks_norm.reset_index(drop=True)

    HRRR_inversion_ranks['Distance'] = HRRR_inversion_ranks_norm['Distance']

    HRRR_inversion_ranks = HRRR_inversion_ranks.sort_values(by='Distance', ascending=False)
    HRRR_inversion_ranks = HRRR_inversion_ranks.reset_index(drop=True)

    # Save the primary inversions and return them
    OBS_prime = OBS_inversion_ranks.head(1)
    HRRR_prime = HRRR_inversion_ranks.head(1)

    return OBS_prime, HRRR_prime


# Check the correctness of the HRRR
def category(OBS_prime, HRRR_prime):
    # First check if either primary inversion vector dataframe is empty
    # find false positives or false negatives
    if (len(OBS_prime) != 0) and (len(HRRR_prime) == 0):
        return 'False Negative'

    elif (len(OBS_prime) == 0) and (len(HRRR_prime) != 0):
        return 'False Positive'

    elif (len(OBS_prime) == 0) and (len(HRRR_prime) == 0):
        return 'No inversions'

    # Now check if the heights overlap
    else:
        # Height domain
        h_min = HRRR_prime['Starting Height'].values[0]
        h_max = HRRR_prime['Ending Height'].values[0]

        # filter out inversions whose domain does not overlap with the primary HRRR inversion
        OBS_filtered = OBS_prime[(OBS_prime['Ending Height'] >= h_min) & (OBS_prime['Starting Height'] <= h_max)]

        # if we filtered out all the inversions, meaning none overlapped with the HRRR inversion, then we can assume the HRRR
        # did not properly detect a legitamate inversion
        if len(OBS_filtered) == 0:
            return 'False Positive'
        else:
            return 'Successful Inversions'


# Start a total results counter:
total_entries_all = []
false_positive_counter_all = []
false_negative_counter_all = []
no_inversion_counter_all = []
successful_inversion_counter_all = []
perc_succ_all = []
perc_succ_inversions_all = []

# Iterate through each station
for i in range(len(stids_HRRR)):

    # Start a results counter for the current station
    false_positive_counter = 0
    false_negative_counter = 0
    no_inversion_counter = 0
    successful_inversion_counter = 0

    # iterate through each date
    curr_time = start_time
    while curr_time <= stop_time:

        date_str = datetime.strftime(curr_time, "%Y-%m-%d %H:%M")

        # Make 10 attempts to grab the modelled dataframe. If 10 attempts fail, we'll pass over this date
        grabbed = False
        for j in range(10):
            print('Grabbing HRRR data for ' + stids_HRRR[i] + '. ATTEMPT ' + str(j + 1) + '/10')
            HRRR_df = mk_HRRR_df(date_str, stids_HRRR[i], sounding_dir)

            if type(HRRR_df) == str:
                if HRRR_df == 'nan':
                    print('Failed to grab HRRR data. Retrying....')
            elif isinstance(HRRR_df, pd.DataFrame):
                grabbed = True
                break
            else:
                print('Failed to grab HRRR data. Retrying....')

        if not grabbed:
            print('Failed to grab HRRR data for ' + date_str + '. Skipping....')
            # jump to next date
            curr_time = curr_time + timedelta(hours=12)
            continue

        # Do the same for the observed data
        grabbed = False
        for j in range(10):
            print('Grabbing OBS data for ' + stids_HRRR[i] + '. ATTEMPT ' + str(j + 1) + '/10')
            OBS_df = get_OBS_data(date_str, stids_HRRR[i], stids_OBS[i], sounding_dir)

            if type(OBS_df) == str:
                if OBS_df == 'nan':
                    print('Failed to grab OBS data. Retrying....')
            elif isinstance(OBS_df, pd.DataFrame):
                grabbed = True
                break
            else:
                print('Failed to grab OBS data. Retrying....')

        if not grabbed:
            print('Failed to grab OBS data for ' + date_str + '. Skipping....')
            # jump to next date
            curr_time = curr_time + timedelta(hours=12)
            continue

        # Now we should have a dataframe for each. Let's now get the primary inversions if any exist
        # Fix the starting heights
        OBS_df, HRRR_df = start_at_same_height(OBS_df, HRRR_df)

        # Convert to standard and normalized height coordinates
        OBS_df_standard, HRRR_df_standard = mk_height_df(OBS_df, HRRR_df)
        OBS_df_normalized, HRRR_df_normalized = mk_normalized_height_df(OBS_df_standard, HRRR_df_standard)

        # get the primary inversions
        OBS_prime, HRRR_prime = inversion_grabber(OBS_df_standard, OBS_df_normalized, HRRR_df_standard,
                                                  HRRR_df_normalized)

        # Finally, let's get our result
        result = category(OBS_prime, HRRR_prime)

        if result == 'False Negative':
            false_negative_counter = false_negative_counter + 1
        elif result == 'False Positive':
            false_positive_counter = false_positive_counter + 1
        elif result == 'No inversions':
            no_inversion_counter = no_inversion_counter + 1
        elif result == 'Successful Inversions':
            successful_inversion_counter = successful_inversion_counter + 1
        else:
            print('FAILED TO DETERMINE THE RESULT!!!!!!!')

        # jump to next date
        curr_time = curr_time + timedelta(hours=12)

    # Calculate the correctness
    total_inversions = successful_inversion_counter + false_negative_counter
    total_entries = total_inversions + false_positive_counter + no_inversion_counter

    # determine the percent correctness
    perc_succ = (float(successful_inversion_counter + no_inversion_counter) / float(total_entries)) * 100
    perc_succ = round(perc_succ, 2)
    perc_succ_inversions = (float(successful_inversion_counter) / float(total_inversions)) * 100
    perc_succ_inversions = round(perc_succ_inversions, 2)

    # save the result to memory
    total_entries_all.append(total_entries)
    successful_inversion_counter_all.append(successful_inversion_counter)
    no_inversion_counter_all.append(no_inversion_counter)
    false_negative_counter_all.append(false_negative_counter)
    false_positive_counter_all.append(false_positive_counter)
    perc_succ_all.append(perc_succ)
    perc_succ_inversions_all.append(perc_succ_inversions)

# print the results for easy reading within jupyter
for i in range(len(stids_HRRR)):
    total_entries = successful_inversion_counter_all[i] + no_inversion_counter_all[i] + false_negative_counter_all[i] + \
                    false_positive_counter_all[i]

    print('\n\n')
    print('########################')
    print('### RESULTS FOR ' + stids_HRRR[i] + ' ###')
    print('########################')
    print('\n')

    print(f"Number of rows checked: {total_entries}")
    print(f"False Positives: {false_positive_counter_all[i]}")
    print(f"False Negatives: {false_negative_counter_all[i]}")
    print(f"No Inversion: {no_inversion_counter_all[i]}")
    print(f"Successful Inversions: {successful_inversion_counter_all[i]}")
    print(f"\nPercent Correct (overall): {perc_succ_all[i]}%")
    print(f"Percent Correct (inversions only): {perc_succ_inversions_all[i]}%")

# save the data
filename = 'correctness_results.csv'

if not os.path.exists('./' + filename):
    # start the file
    header = ['Cutoff Height (m)', '',
              'TOTAL - Total number of soundings', 'TOTAL - Successful inversion count', 'TOTAL - No inversion count',
              'TOTAL - False positive count', 'TOTAL - False negative count', 'TOTAL - Percent Successful (overall)',
              'TOTAL - Percent Successful (inversions only)', '',
              'KUNR - Total number of soundings', 'KUNR - Successful inversion count', 'KUNR - No inversion count',
              'KUNR - False positive count', 'KUNR - False negative count', 'KUNR - Percent Successful (overall)',
              'KUNR - Percent Successful (inversions only)', '',
              'KRIW - Total number of soundings', 'KRIW - Successful inversion count', 'KRIW - No inversion count',
              'KRIW - False positive count', 'KRIW - False negative count', 'KRIW - Percent Successful (overall)',
              'KRIW - Percent Successful (inversions only)', '',
              'KGGW - Total number of soundings', 'KGGW - Successful inversion count', 'KGGW - No inversion count',
              'KGGW - False positive count', 'KGGW - False negative count', 'KGGW - Percent Successful (overall)',
              'KGGW - Percent Successful (inversions only)', '',
              'KDNR - Total number of soundings', 'KDNR - Successful inversion count', 'KDNR - No inversion count',
              'KDNR - False positive count', 'KDNR - False negative count', 'KDNR - Percent Successful (overall)',
              'KDNR - Percent Successful (inversions only)', ''
              ]

    with open('./' + filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)

# get total stats
false_positive_counter = 0
false_negative_counter = 0
no_inversion_counter = 0
successful_inversion_counter = 0

for i in range(len(perc_succ_all)):
    false_positive_counter = false_positive_counter + false_positive_counter_all[i]
    false_negative_counter = false_negative_counter + false_negative_counter_all[i]
    no_inversion_counter = no_inversion_counter + no_inversion_counter_all[i]
    successful_inversion_counter = successful_inversion_counter + successful_inversion_counter_all[i]

# Calculate the correctness
total_inversions = successful_inversion_counter + false_negative_counter
total_entries = total_inversions + false_positive_counter + no_inversion_counter

# determine the percent correctness
perc_succ = (float(successful_inversion_counter + no_inversion_counter) / float(total_entries)) * 100
perc_succ = round(perc_succ, 2)
perc_succ_inversions = (float(successful_inversion_counter) / float(total_inversions)) * 100
perc_succ_inversions = round(perc_succ_inversions, 2)

# print the results

print('\n\n')
print('################################')
print('### RESULTS FOR ALL STATIONS ###')
print('################################')
print('\n')

print(f"Number of rows checked: {total_entries}")
print(f"False Positives: {false_positive_counter}")
print(f"False Negatives: {false_negative_counter}")
print(f"No Inversion: {no_inversion_counter}")
print(f"Successful Inversions: {successful_inversion_counter}")
print(f"\nPercent Correct (overall): {perc_succ}%")
print(f"Percent Correct (inversions only): {perc_succ_inversions}%")

# Save the data
row = [5000, '',
       total_entries, successful_inversion_counter, no_inversion_counter, false_positive_counter,
       false_negative_counter, perc_succ, perc_succ_inversions, '',

       total_entries_all[0], successful_inversion_counter_all[0], no_inversion_counter_all[0],
       false_positive_counter_all[0], false_negative_counter_all[0], perc_succ_all[0], perc_succ_inversions_all[0], '',
       total_entries_all[1], successful_inversion_counter_all[1], no_inversion_counter_all[1],
       false_positive_counter_all[1], false_negative_counter_all[1], perc_succ_all[1], perc_succ_inversions_all[1], '',
       total_entries_all[2], successful_inversion_counter_all[2], no_inversion_counter_all[2],
       false_positive_counter_all[2], false_negative_counter_all[2], perc_succ_all[2], perc_succ_inversions_all[2], '',
       total_entries_all[3], successful_inversion_counter_all[3], no_inversion_counter_all[3],
       false_positive_counter_all[3], false_negative_counter_all[3], perc_succ_all[3], perc_succ_inversions_all[3], '']

with open('./' + filename, 'a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(row)

# set a cutoff height we haven't done yet
cutoff = 4900

# iteratively reduce the cutoff height. From trial and error, going below 500m produces too few inversions to be worthwhile
while cutoff >= 500:

    # get stats for each station
    # Start a total results counter:
    total_entries_all = []
    false_positive_counter_all = []
    false_negative_counter_all = []
    no_inversion_counter_all = []
    successful_inversion_counter_all = []
    perc_succ_all = []
    perc_succ_inversions_all = []

    # Iterate through each station
    for i in range(len(stids_HRRR)):

        # Start a results counter for the current station
        false_positive_counter = 0
        false_negative_counter = 0
        no_inversion_counter = 0
        successful_inversion_counter = 0

        # iterate through each date
        curr_time = start_time
        while curr_time <= stop_time:

            date_str = datetime.strftime(curr_time, "%Y-%m-%d %H:%M")

            # Make 10 attempts to grab the modelled dataframe. If 10 attempts fail, we'll pass over this date
            grabbed = False
            for j in range(10):
                print('Grabbing HRRR data for ' + stids_HRRR[i] + '. ATTEMPT ' + str(j + 1) + '/10')
                HRRR_df = mk_HRRR_df(date_str, stids_HRRR[i], sounding_dir)

                if type(HRRR_df) == str:
                    if HRRR_df == 'nan':
                        print('Failed to grab HRRR data. Retrying....')
                elif isinstance(HRRR_df, pd.DataFrame):
                    grabbed = True
                    break
                else:
                    print('Failed to grab HRRR data. Retrying....')

            if not grabbed:
                print('Failed to grab HRRR data for ' + date_str + '. Skipping....')
                # jump to next date
                curr_time = curr_time + timedelta(hours=12)
                continue

            # Do the same for the observed data
            grabbed = False
            for j in range(10):
                print('Grabbing OBS data for ' + stids_HRRR[i] + '. ATTEMPT ' + str(j + 1) + '/10')
                OBS_df = get_OBS_data(date_str, stids_HRRR[i], stids_OBS[i], sounding_dir)

                if type(OBS_df) == str:
                    if OBS_df == 'nan':
                        print('Failed to grab OBS data. Retrying....')
                elif isinstance(OBS_df, pd.DataFrame):
                    grabbed = True
                    break
                else:
                    print('Failed to grab OBS data. Retrying....')

            if not grabbed:
                print('Failed to grab OBS data for ' + date_str + '. Skipping....')
                # jump to next date
                curr_time = curr_time + timedelta(hours=12)
                continue

            # Now we should have a dataframe for each. Let's now get the primary inversions if any exist
            # Fix the starting heights
            OBS_df, HRRR_df = start_at_same_height(OBS_df, HRRR_df)

            # Convert to standard and normalized height coordinates
            OBS_df_standard, HRRR_df_standard = mk_height_df(OBS_df, HRRR_df, cutoff=cutoff)
            OBS_df_normalized, HRRR_df_normalized = mk_normalized_height_df(OBS_df_standard, HRRR_df_standard)

            # get the primary inversions
            OBS_prime, HRRR_prime = inversion_grabber(OBS_df_standard, OBS_df_normalized, HRRR_df_standard,
                                                      HRRR_df_normalized)

            # Finally, let's get our result
            result = category(OBS_prime, HRRR_prime)

            if result == 'False Negative':
                false_negative_counter = false_negative_counter + 1
            elif result == 'False Positive':
                false_positive_counter = false_positive_counter + 1
            elif result == 'No inversions':
                no_inversion_counter = no_inversion_counter + 1
            elif result == 'Successful Inversions':
                successful_inversion_counter = successful_inversion_counter + 1
            else:
                print('FAILED TO DETERMINE THE RESULT!!!!!!!')

            # jump to next date
            curr_time = curr_time + timedelta(hours=12)

        # Calculate the correctness
        total_inversions = successful_inversion_counter + false_negative_counter
        total_entries = total_inversions + false_positive_counter + no_inversion_counter

        # determine the percent correctness
        perc_succ = (float(successful_inversion_counter + no_inversion_counter) / float(total_entries)) * 100
        perc_succ = round(perc_succ, 2)
        perc_succ_inversions = (float(successful_inversion_counter) / float(total_inversions)) * 100
        perc_succ_inversions = round(perc_succ_inversions, 2)

        # save the result to memory
        total_entries_all.append(total_entries)
        successful_inversion_counter_all.append(successful_inversion_counter)
        no_inversion_counter_all.append(no_inversion_counter)
        false_negative_counter_all.append(false_negative_counter)
        false_positive_counter_all.append(false_positive_counter)
        perc_succ_all.append(perc_succ)
        perc_succ_inversions_all.append(perc_succ_inversions)

    # get total stats
    false_positive_counter = 0
    false_negative_counter = 0
    no_inversion_counter = 0
    successful_inversion_counter = 0

    for i in range(len(perc_succ_all)):
        false_positive_counter = false_positive_counter + false_positive_counter_all[i]
        false_negative_counter = false_negative_counter + false_negative_counter_all[i]
        no_inversion_counter = no_inversion_counter + no_inversion_counter_all[i]
        successful_inversion_counter = successful_inversion_counter + successful_inversion_counter_all[i]

    # Calculate the correctness
    total_inversions = successful_inversion_counter + false_negative_counter
    total_entries = total_inversions + false_positive_counter + no_inversion_counter

    # determine the percent correctness
    perc_succ = (float(successful_inversion_counter + no_inversion_counter) / float(total_entries)) * 100
    perc_succ = round(perc_succ, 2)
    perc_succ_inversions = (float(successful_inversion_counter) / float(total_inversions)) * 100
    perc_succ_inversions = round(perc_succ_inversions, 2)

    # Save the data
    row = [cutoff, '',
           total_entries, successful_inversion_counter, no_inversion_counter, false_positive_counter,
           false_negative_counter, perc_succ, perc_succ_inversions, '',

           total_entries_all[0], successful_inversion_counter_all[0], no_inversion_counter_all[0],
           false_positive_counter_all[0], false_negative_counter_all[0], perc_succ_all[0], perc_succ_inversions_all[0],
           '',
           total_entries_all[1], successful_inversion_counter_all[1], no_inversion_counter_all[1],
           false_positive_counter_all[1], false_negative_counter_all[1], perc_succ_all[1], perc_succ_inversions_all[1],
           '',
           total_entries_all[2], successful_inversion_counter_all[2], no_inversion_counter_all[2],
           false_positive_counter_all[2], false_negative_counter_all[2], perc_succ_all[2], perc_succ_inversions_all[2],
           '',
           total_entries_all[3], successful_inversion_counter_all[3], no_inversion_counter_all[3],
           false_positive_counter_all[3], false_negative_counter_all[3], perc_succ_all[3], perc_succ_inversions_all[3],
           '']

    with open('./' + filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(row)

    cutoff = cutoff - 100

# setup mesonet station directories, in line with the stids_HRRR and stids_OBS lists
mesonet_stids = ['KUNR', 'KRIW', 'KGGW', 'URBC2']
mesonet_dir = './mesonet_csvs/'

# Grab Mesonet Data

mesonet_dfs = []

for stid in mesonet_stids:
    dates = []
    temps = []

    mesonet_csv = mesonet_dir + stid + '.csv'

    with open(mesonet_csv) as f:
        data = f.read()

    # parse out the header
    lines = data.split("\n")
    header = lines[10].split(",")
    lines = lines[12:]

    for i in range(len(lines) - 1):
        values = lines[i].split(",")

        if values[1] != '':
            dates.append(datetime.strptime(values[1], "%Y-%m-%dT%H:%M:%SZ"))

            if values[2] != '':
                temps.append(float(values[2]))
            else:
                temps.append(np.nan)

    mesonet_data = pd.DataFrame({
        'Dates': dates,
        'Temperature': temps
    })
    mesonet_data = mesonet_data.dropna(subset=['Temperature'])
    mesonet_data = mesonet_data.reset_index(drop=True)

    mesonet_dfs.append(mesonet_data)

# start a new csv file
filename = 'correctness_results_mesonet_applied.csv'

if not os.path.exists('./' + filename):
    # start the file
    header = ['Cutoff Height (m)', '',
              'TOTAL - Total number of soundings', 'TOTAL - Successful inversion count', 'TOTAL - No inversion count',
              'TOTAL - False positive count', 'TOTAL - False negative count', 'TOTAL - Percent Successful (overall)',
              'TOTAL - Percent Successful (inversions only)', '',
              'KUNR - Total number of soundings', 'KUNR - Successful inversion count', 'KUNR - No inversion count',
              'KUNR - False positive count', 'KUNR - False negative count', 'KUNR - Percent Successful (overall)',
              'KUNR - Percent Successful (inversions only)', '',
              'KRIW - Total number of soundings', 'KRIW - Successful inversion count', 'KRIW - No inversion count',
              'KRIW - False positive count', 'KRIW - False negative count', 'KRIW - Percent Successful (overall)',
              'KRIW - Percent Successful (inversions only)', '',
              'KGGW - Total number of soundings', 'KGGW - Successful inversion count', 'KGGW - No inversion count',
              'KGGW - False positive count', 'KGGW - False negative count', 'KGGW - Percent Successful (overall)',
              'KGGW - Percent Successful (inversions only)', '',
              'KDNR - Total number of soundings', 'KDNR - Successful inversion count', 'KDNR - No inversion count',
              'KDNR - False positive count', 'KDNR - False negative count', 'KDNR - Percent Successful (overall)',
              'KDNR - Percent Successful (inversions only)', ''
              ]

    with open('./' + filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)

# set the starting cutoff height
cutoff = 5000

# iteratively reduce the cutoff height. From trial and error, going below 500m produces too few inversions to be worthwhile
while cutoff >= 500:

    # get stats for each station
    # Start a total results counter:
    total_entries_all = []
    false_positive_counter_all = []
    false_negative_counter_all = []
    no_inversion_counter_all = []
    successful_inversion_counter_all = []
    perc_succ_all = []
    perc_succ_inversions_all = []

    # Iterate through each station
    for i in range(len(stids_HRRR)):

        # Start a results counter for the current station
        false_positive_counter = 0
        false_negative_counter = 0
        no_inversion_counter = 0
        successful_inversion_counter = 0

        # iterate through each date
        curr_time = start_time
        while curr_time <= stop_time:

            date_str = datetime.strftime(curr_time, "%Y-%m-%d %H:%M")

            # Make 10 attempts to grab the modelled dataframe. If 10 attempts fail, we'll pass over this date
            grabbed = False
            for j in range(10):
                print('Grabbing HRRR data for ' + stids_HRRR[i] + '. ATTEMPT ' + str(j + 1) + '/10')
                HRRR_df = mk_HRRR_df(date_str, stids_HRRR[i], sounding_dir)

                if type(HRRR_df) == str:
                    if HRRR_df == 'nan':
                        print('Failed to grab HRRR data. Retrying....')
                elif isinstance(HRRR_df, pd.DataFrame):
                    grabbed = True
                    break
                else:
                    print('Failed to grab HRRR data. Retrying....')

            if not grabbed:
                print('Failed to grab HRRR data for ' + date_str + '. Skipping....')
                # jump to next date
                curr_time = curr_time + timedelta(hours=12)
                continue

            # Do the same for the observed data
            grabbed = False
            for j in range(10):
                print('Grabbing OBS data for ' + stids_HRRR[i] + '. ATTEMPT ' + str(j + 1) + '/10')
                OBS_df = get_OBS_data(date_str, stids_HRRR[i], stids_OBS[i], sounding_dir)

                if type(OBS_df) == str:
                    if OBS_df == 'nan':
                        print('Failed to grab OBS data. Retrying....')
                elif isinstance(OBS_df, pd.DataFrame):
                    grabbed = True
                    break
                else:
                    print('Failed to grab OBS data. Retrying....')

            if not grabbed:
                print('Failed to grab OBS data for ' + date_str + '. Skipping....')
                # jump to next date
                curr_time = curr_time + timedelta(hours=12)
                continue

            # Now we should have a dataframe for each. Let's now get the primary inversions if any exist
            # Fix the starting heights
            OBS_df, HRRR_df = start_at_same_height(OBS_df, HRRR_df)

            #####################################################
            ###########     APPLYING MESONET DATA     ###########
            #####################################################

            # Grab the data
            mesonet_df = mesonet_dfs[i]
            sfc_temp = None

            # if the closest time in the mesonet data exceeds 1 hour, just consider that a failure
            max_time_difference = timedelta(hours=1)

            # Find the closest date
            closest_date = mesonet_df.iloc[(mesonet_df['Dates'] - curr_time).abs().idxmin()]['Dates']

            # Check if the time difference is within the specified limit
            if abs(closest_date - curr_time) < max_time_difference:
                sfc_temp = mesonet_df.iloc[(mesonet_df['Dates'] - curr_time).abs().idxmin()]['Temperature']

            # replace the sfc value on the HRRR if we found one in the mesonet data
            if sfc_temp is not None:
                HRRR_df['Temp (°C)'].iloc[0] = sfc_temp

            # Convert to standard and normalized height coordinates
            OBS_df_standard, HRRR_df_standard = mk_height_df(OBS_df, HRRR_df, cutoff=cutoff)
            OBS_df_normalized, HRRR_df_normalized = mk_normalized_height_df(OBS_df_standard, HRRR_df_standard)

            # get the primary inversions
            OBS_prime, HRRR_prime = inversion_grabber(OBS_df_standard, OBS_df_normalized, HRRR_df_standard,
                                                      HRRR_df_normalized)

            # Finally, let's get our result
            result = category(OBS_prime, HRRR_prime)

            if result == 'False Negative':
                false_negative_counter = false_negative_counter + 1
            elif result == 'False Positive':
                false_positive_counter = false_positive_counter + 1
            elif result == 'No inversions':
                no_inversion_counter = no_inversion_counter + 1
            elif result == 'Successful Inversions':
                successful_inversion_counter = successful_inversion_counter + 1
            else:
                print('FAILED TO DETERMINE THE RESULT!!!!!!!')

            # jump to next date
            curr_time = curr_time + timedelta(hours=12)

        # Calculate the correctness
        total_inversions = successful_inversion_counter + false_negative_counter
        total_entries = total_inversions + false_positive_counter + no_inversion_counter

        # determine the percent correctness
        perc_succ = (float(successful_inversion_counter + no_inversion_counter) / float(total_entries)) * 100
        perc_succ = round(perc_succ, 2)
        perc_succ_inversions = (float(successful_inversion_counter) / float(total_inversions)) * 100
        perc_succ_inversions = round(perc_succ_inversions, 2)

        # save the result to memory
        total_entries_all.append(total_entries)
        successful_inversion_counter_all.append(successful_inversion_counter)
        no_inversion_counter_all.append(no_inversion_counter)
        false_negative_counter_all.append(false_negative_counter)
        false_positive_counter_all.append(false_positive_counter)
        perc_succ_all.append(perc_succ)
        perc_succ_inversions_all.append(perc_succ_inversions)

    # get total stats
    false_positive_counter = 0
    false_negative_counter = 0
    no_inversion_counter = 0
    successful_inversion_counter = 0

    for i in range(len(perc_succ_all)):
        false_positive_counter = false_positive_counter + false_positive_counter_all[i]
        false_negative_counter = false_negative_counter + false_negative_counter_all[i]
        no_inversion_counter = no_inversion_counter + no_inversion_counter_all[i]
        successful_inversion_counter = successful_inversion_counter + successful_inversion_counter_all[i]

    # Calculate the correctness
    total_inversions = successful_inversion_counter + false_negative_counter
    total_entries = total_inversions + false_positive_counter + no_inversion_counter

    # determine the percent correctness
    perc_succ = (float(successful_inversion_counter + no_inversion_counter) / float(total_entries)) * 100
    perc_succ = round(perc_succ, 2)
    perc_succ_inversions = (float(successful_inversion_counter) / float(total_inversions)) * 100
    perc_succ_inversions = round(perc_succ_inversions, 2)

    # Save the data
    row = [cutoff, '',
           total_entries, successful_inversion_counter, no_inversion_counter, false_positive_counter,
           false_negative_counter, perc_succ, perc_succ_inversions, '',

           total_entries_all[0], successful_inversion_counter_all[0], no_inversion_counter_all[0],
           false_positive_counter_all[0], false_negative_counter_all[0], perc_succ_all[0], perc_succ_inversions_all[0],
           '',
           total_entries_all[1], successful_inversion_counter_all[1], no_inversion_counter_all[1],
           false_positive_counter_all[1], false_negative_counter_all[1], perc_succ_all[1], perc_succ_inversions_all[1],
           '',
           total_entries_all[2], successful_inversion_counter_all[2], no_inversion_counter_all[2],
           false_positive_counter_all[2], false_negative_counter_all[2], perc_succ_all[2], perc_succ_inversions_all[2],
           '',
           total_entries_all[3], successful_inversion_counter_all[3], no_inversion_counter_all[3],
           false_positive_counter_all[3], false_negative_counter_all[3], perc_succ_all[3], perc_succ_inversions_all[3],
           '']

    with open('./' + filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(row)

    cutoff = cutoff - 100