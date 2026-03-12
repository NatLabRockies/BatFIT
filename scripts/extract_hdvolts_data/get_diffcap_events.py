from file_management import *
from read import *
from prettyPlot.plotting import *
from batfit import logger
#logger.setLevel("DEBUG")
logger.setLevel("INFO")

file_names = get_all_protocol_all_rpt_data_file(data_type="diffcap")

step_1_amp = None
step_3_amp = None

for protocol in file_names:
    for rpt_id in file_names[protocol]:
        for cell_in in file_names[protocol][rpt_id]:
            filename = file_names[protocol][rpt_id][cell_in]
            cycle_df = read_single_csv(filename, data_type="diffcap")
            list_of_dfs = break_by_contiguous_step(cycle_df)
            assert len(list_of_dfs) == 4
            if step_1_amp is None:
                step_1_amp = get_current(list_of_dfs[0])[1:]
            else:
                step_1_amp = np.hstack((step_1_amp, get_current(list_of_dfs[0])[1:]))
            if step_3_amp is None:
                step_3_amp = get_current(list_of_dfs[2])[1:]
            else:
                step_3_amp = np.hstack((step_3_amp, get_current(list_of_dfs[2])[1:]))

print(f"CC Charge = {np.mean(step_1_amp):.10g}A +/- {np.std(step_1_amp):.5g}A ")
print(f"CC Discharge = {np.mean(step_3_amp):.10g}A +/- {np.std(step_3_amp):.5g}A ")

