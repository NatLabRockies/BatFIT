from file_management import *
from read import *
from prettyPlot.plotting import *
from batfit import logger
#logger.setLevel("DEBUG")
logger.setLevel("INFO")

file_names = get_all_protocol_all_rpt_data_file(data_type="posthppc")
step_values = [13,16,18,19,21,23]

step_data = {}
for step in step_values:
    step_data[step] = {"time": None, "amp": None, "finAh": None, "ampcc": None}


for protocol in file_names:
    for rpt_id in file_names[protocol]:
        for cell_in in file_names[protocol][rpt_id]:
            filename = file_names[protocol][rpt_id][cell_in]
            cycle_df = read_single_csv(filename, data_type="posthppc")
            list_of_dfs = break_by_contiguous_step(cycle_df)
            step_dfs = [int(df["Step"].mean()) for df in list_of_dfs]

            for istep, step_df in enumerate(step_dfs):
                voltage = get_voltage(list_of_dfs[istep])
                current = get_current(list_of_dfs[istep])[1:]
                duration = get_duration(list_of_dfs[istep])
                finAh = get_final_Ah(list_of_dfs[istep])
                for step_value in step_values:
                    if step_df == step_value:
                        if step_data[step_value]["time"] is None:
                            step_data[step_value]["time"] = duration
                        else:
                            step_data[step_value]["time"] = np.hstack((step_data[step_value]["time"], duration))
                        if step_data[step_value]["amp"] is None:
                            step_data[step_value]["amp"] = current
                        else:
                            step_data[step_value]["amp"] = np.hstack((step_data[step_value]["amp"], current))
                        if step_data[step_value]["finAh"] is None:
                            step_data[step_value]["finAh"] = finAh
                        else:
                            step_data[step_value]["finAh"] = np.hstack((step_data[step_value]["finAh"], finAh))

                        mask = voltage.between(3, 4)
                        filtered_current = current[mask].copy()
                        if step_data[step_value]["ampcc"] is None:
                            step_data[step_value]["ampcc"] = filtered_current
                        else:
                            step_data[step_value]["ampcc"] = np.hstack((step_data[step_value]["ampcc"], filtered_current))

for step_value in step_values:
    curr = step_data[step_value]["amp"]
    currcc = step_data[step_value]["ampcc"]
    dur = step_data[step_value]["time"] 
    finAh = step_data[step_value]["finAh"] 
    print(f"STEP {step_value}")
    print(f"\tAMP = {np.mean(curr):.10g} A +/- {np.std(curr):.5g} A ")
    print(f"\tAMP CC = {np.mean(currcc):.10g} A +/- {np.std(currcc):.5g} A ")
    print(f"\tTime = {np.mean(dur):.10g} min +/- {np.std(dur):.5g} min ")
    print(f"\tfinAh = {np.mean(finAh):.10g} Ah +/- {np.std(finAh):.5g} Ah ")

