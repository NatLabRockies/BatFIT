from file_management import *
from read import *
from prettyPlot.plotting import *
from batfit import logger
#logger.setLevel("DEBUG")
logger.setLevel("INFO")

file_names = get_all_protocol_all_rpt_data_file(data_type="posthppc")

step_1_amp = None
step_3_time = None


for protocol in file_names:
    for rpt_id in file_names[protocol]:
        for cell_in in file_names[protocol][rpt_id]:
            filename = file_names[protocol][rpt_id][cell_in]
            cycle_df = read_single_csv(filename, data_type="posthppc")
            list_of_dfs = break_by_step(cycle_df)
            if len(list_of_dfs) == 11:
                print(f"######### {rpt_id}")
            print([int(df["Step"].mean()) for df in list_of_dfs])
            try:
                #assert len(list_of_dfs) == 14
                assert len(list_of_dfs)>8
            except AssertionError:
                print(filename)
                breakpoint()
            if step_1_amp is None:
                step_1_amp = get_current(list_of_dfs[0])[1:]
            else:
                step_1_amp = np.hstack((step_1_amp, get_current(list_of_dfs[0])[1:]))
            if step_3_time is None:
                step_3_time = get_duration(list_of_dfs[2])
            else:
                step_3_time = np.hstack((step_3_time, get_duration(list_of_dfs[2])))

print(f"STEP 1 CC Charge = {np.mean(step_1_amp):.10g}A +/- {np.std(step_1_amp):.5g}A ")
print(f"STEP 3 Rest Time = {np.mean(step_3_time):.10g} min +/- {np.std(step_3_time):.5g} min ")

