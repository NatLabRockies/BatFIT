from file_management import *
from read import *
from prettyPlot.plotting import *
from batfit import logger
#logger.setLevel("DEBUG")
logger.setLevel("INFO")

file_names = get_all_protocol_all_rpt_data_file(data_type="hppc")

step_1_amp = None
step_3_time = None
step_4_amp = None
step_5_time = None
step_6_amp = None
step_8_time = None


for protocol in file_names:
    for rpt_id in file_names[protocol]:
        for cell_in in file_names[protocol][rpt_id]:
            filename = file_names[protocol][rpt_id][cell_in]
            cycle_df = read_single_csv(filename, data_type="hppc")
            list_of_dfs = break_by_contiguous_step(cycle_df)
            try:
                assert len(list_of_dfs)>8
            except AssertionError:
                print(filename)
                breakpoint()
            if step_1_amp is None:
                step_1_amp = get_current(list_of_dfs[0])[1:]
            else:
                step_1_amp = np.hstack((step_1_amp, get_current(list_of_dfs[0])[1:]))
            if abs(get_duration(list_of_dfs[2])-60)>1:
                print(get_duration(list_of_dfs[2]))
                print(list_of_dfs[2]["Step"].mean())
                print(filename)
                breakpoint()
            if step_3_time is None:
                step_3_time = get_duration(list_of_dfs[2])
            else:
                step_3_time = np.hstack((step_3_time, get_duration(list_of_dfs[2])))
            if step_4_amp is None:
                step_4_amp = get_current(list_of_dfs[3])[1:]
            else:
                step_4_amp = np.hstack((step_4_amp, get_current(list_of_dfs[3])[1:]))
            if step_5_time is None:
                step_5_time = get_duration(list_of_dfs[4])
            else:
                step_5_time = np.hstack((step_5_time, get_duration(list_of_dfs[4])))

print(f"STEP 1 CC Charge = {np.mean(step_1_amp):.10g}A +/- {np.std(step_1_amp):.5g}A ")
print(f"STEP 3 Rest Time = {np.mean(step_3_time):.10g} min +/- {np.std(step_3_time):.5g} min ")

print(f"STEP 4 CC Discharge = {np.mean(step_4_amp):.10g}A +/- {np.std(step_4_amp):.5g}A ")
print(f"STEP 5 Rest Time = {np.mean(step_5_time):.10g} min +/- {np.std(step_5_time):.5g} min ")
