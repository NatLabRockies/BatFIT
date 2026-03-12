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

def remove_rogue_twos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes singular 'rogue' records where the step value is 2 
    (i.e., a 2 surrounded by non-2 values).
    """
    is_two = df["Step"] == 2
    
    prev_is_not_two = df["Step"].shift(1) != 2
    next_is_not_two = df["Step"].shift(-1) != 2
    
    is_rogue = is_two & prev_is_not_two & next_is_not_two
    
    clean_df = df[~is_rogue].copy()
    
    return clean_df

def apply_hppc_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes all rows before the first occurrence of 'C' in a specific column.
    """
    df = remove_rogue_twos(df)
    try:
        is_c = df["State"] == 'C'
    except KeyError:
        is_c = df["MD"] == 'C'
    first_c_position = is_c.argmax()
    filtered_df = df.iloc[first_c_position:].copy()
    return filtered_df

def apply_post_hppc_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes all rows before the first occurrence of 'C' in a specific column.
    """
    filtered_df = apply_hppc_filter(df)
    filtered_df = filtered_df[filtered_df["Step"] >= 12].copy()
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
    
    # Remove entries for which test Times is redundant
    mask = diff_time.isna() | (diff_time >= 1e-6)
    filtered_df = raw[mask].copy()
    num_removed = len(raw) - len(filtered_df)
    
    # Remove entries for which the step id is non increasing
    #mask = filtered_df["Step"] >= filtered_df["Step"].cummax()
    # Apply the mask and return a clean, independent copy
    #filtered_df = filtered_df[mask].copy()

    if data_type is not None:
       if data_type.lower() == "diffcap":
           filtered_df = apply_diffcap_filter(filtered_df)
       if data_type.lower() == "hppc":
           filtered_df = apply_hppc_filter(filtered_df)
       if data_type.lower() == "posthppc":
           filtered_df = apply_post_hppc_filter(filtered_df)

    logger.info(f"Read {fpath} ({filtered_df.shape})")

    # Return the Pandas DataFrame, not a numpy array
    return filtered_df

def break_by_step(df: pd.DataFrame) -> list:
    """
    Breaks a DataFrame into a list of DataFrames based on the 'Step' column.
    """
    # Safety check
    if "Step" not in df.columns:
        raise ValueError("The column 'Step' was not found in the DataFrame.")
    
    list_of_dfs = [group_df for step_name, group_df in df.groupby("Step", sort=False)]
    
    return list_of_dfs

def break_by_contiguous_step(df: pd.DataFrame) -> list:
    """
    Breaks a DataFrame into a list of DataFrames based on contiguous blocks 
    of the 'Step' column.
    """
    # Safety check
    if "Step" not in df.columns:
        raise ValueError("The column 'Step' was not found in the DataFrame.")

    step_changes = df["Step"] != df["Step"].shift()
    # Use cumsum() to create a unique ID for each contiguous block
    # True evaluates to 1, False to 0, so the sum goes up only when the step changes
    block_ids = step_changes.cumsum()
    # Group by this new block_id instead of the literal "Step" value
    list_of_dfs = [group_df.copy() for _, group_df in df.groupby(block_ids)]

    return list_of_dfs



def get_test_time(df):
    try:
        test_time = df["TestTime"]
    except KeyError:
        test_time = df["Test Time (min)"]
    return test_time

def get_elapsed_test_time(df):
    test_time = get_test_time(df)
    return  test_time - test_time.iloc[0]

def get_duration(df):
    elapsed_time = get_elapsed_test_time(df)
    return  elapsed_time.iloc[-1]

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

