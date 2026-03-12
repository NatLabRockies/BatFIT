import numpy as np
import pandas as pd
from prettyPlot.plotting import *
from batfit import logger

def apply_diffcap_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes all rows before the first occurrence of 'C' in a specific column.
    """
    try:
        is_c = df["State"] == 'C'
    except KeyError:
        is_c = df["MD"] == 'C'
    first_c_position = is_c.argmax()
    filtered_df = df.iloc[first_c_position:].copy()
    return filtered_df


def read_single_csv(fpath:str="60018015 - 018.csv", data_type:str|None=None)-> pd.DataFrame:
    """
    Read CSV file into a numpy array
    """
    raw = pd.read_csv(fpath, skiprows=2)

    # Calculate difference between consecutive rows in 'TestTime'
    # The first row will be NaN (no previous row to subtract)
    try:
        diff_time = raw["TestTime"].diff()
    except KeyError:
        diff_time = raw["Test Time (min)"].diff()
    except:
        print(fpath)
        breakpoint()
    # Create a mask: Keep the first row (isna) AND rows where diff >= 1e-6
    mask = diff_time.isna() | (diff_time >= 1e-6)

    # Apply the mask to filter the DataFrame
    filtered_df = raw[mask].copy()

    # Logging logic
    num_removed = len(raw) - len(filtered_df)
    if num_removed > 0:
        logger.debug(f"Removed {num_removed} entries")

    if data_type is not None:
       if data_type.lower() == "diffcap":
           filtered_df = apply_diffcap_filter(filtered_df)

    logger.info(f"Read {fpath} ({filtered_df.shape})")

    # Return the Pandas DataFrame, not a numpy array
    return filtered_df

def get_test_time(df):
    try:
        test_time = df["TestTime"]
    except KeyError:
        test_time = df["Test Time (min)"]
    return test_time

def get_elapsed_test_time(df):
    test_time = get_test_time(df)
    return  test_time - test_time.iloc[0]

def get_current(df):
    try:
        current = df["Amps"]
    except KeyError:
        current = df["Current"]
    return current

def get_voltage(df):
    try:
        voltage = df["Volts"]
    except KeyError:
        voltage = df["Voltage"]
    return voltage

def get_temperature(df):
    temperature = df["Temp"]
    return temperature

#def extract_diff_cap(A:np.ndarray, cyc_id:int):
#    """
#    Extract data from csv by cycle
#    """
#    # If negative cycle, output everything
#    if cyc_id<0:
#         ind = np.array(list(range(A.shape[0]))).reshape((-1,1))
#    else:
#         ind = np.argwhere(A[:,1] == cyc_id)
#    test_time = A[ind,3])
#    step_time = A[ind,4])
#    cap_Ah = remove_redundant_record(ind_remove,A[ind,5])
#    e_Wh = remove_redundant_record(ind_remove,A[ind,6])
#    curr_A = remove_redundant_record(ind_remove,A[ind,7])
#    phi_V = remove_redundant_record(ind_remove,A[ind,8])
#    record = remove_redundant_record(ind_remove,A[ind,0])
#    temp=remove_redundant_record(ind_remove,A[ind,13])
#    try:
#        power_W = np.gradient(e_Wh[:,0],test_time[:,0]/60)
#    except:
#        breakpoint()
#        find_redundant_record(test_time)
#        breakpoint()
#    return {"record_id": record,"test_time":test_time, "step_time":step_time, "cap_Ah":cap_Ah, "e_Wh": e_Wh, "curr_A": curr_A, "phi_V":phi_V, "temp_C": temp, "power_W": power_W}
#

