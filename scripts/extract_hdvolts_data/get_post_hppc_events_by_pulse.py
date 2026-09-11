from file_management import *
from prettyPlot.plotting import *
from read import *

from batfit import logger

# logger.setLevel("DEBUG")
logger.setLevel("INFO")

file_names = get_all_protocol_all_rpt_data_file(data_type="posthppc")
step_values = [13, 14, 16, 18, 19, 21, 23]
step_counter = {}


step_data = {}
for step in step_values:
    step_data[step] = {}
    if step in [13, 14]:
        step_data[step][1] = {
            "time": None,
            "amp": None,
            "finAh": None,
            "ampcc": None,
        }
    if step in [16, 18, 19, 21, 23]:
        for i in range(7):
            step_data[step][i + 1] = {
                "time": None,
                "amp": None,
                "finAh": None,
                "ampcc": None,
            }


for protocol in file_names:
    for rpt_id in file_names[protocol]:
        for cell_in in file_names[protocol][rpt_id]:
            filename = file_names[protocol][rpt_id][cell_in]
            cycle_df = read_single_csv(filename, data_type="posthppc")
            list_of_dfs = break_by_contiguous_step(cycle_df)
            step_dfs = [int(df["Step"].mean()) for df in list_of_dfs]
            for step_value in step_values:
                step_counter[step_value] = 0

            for istep, step_df in enumerate(step_dfs):
                step_counter[step_df] += 1
                counter = step_counter[step_df]
                voltage = get_voltage(list_of_dfs[istep])
                current = get_current(list_of_dfs[istep])[1:]
                duration = get_duration(list_of_dfs[istep])
                finAh = get_final_Ah(list_of_dfs[istep])
                for step_value in step_values:
                    if step_df == step_value:
                        if step_data[step_value][counter]["time"] is None:
                            step_data[step_value][counter]["time"] = duration
                        else:
                            step_data[step_value][counter]["time"] = np.hstack(
                                (
                                    step_data[step_value][counter]["time"],
                                    duration,
                                )
                            )
                        if step_data[step_value][counter]["amp"] is None:
                            step_data[step_value][counter]["amp"] = current
                        else:
                            step_data[step_value][counter]["amp"] = np.hstack(
                                (
                                    step_data[step_value][counter]["amp"],
                                    current,
                                )
                            )
                        if step_data[step_value][counter]["finAh"] is None:
                            step_data[step_value][counter]["finAh"] = finAh
                        else:
                            step_data[step_value][counter]["finAh"] = (
                                np.hstack(
                                    (
                                        step_data[step_value][counter][
                                            "finAh"
                                        ],
                                        finAh,
                                    )
                                )
                            )

                        mask = voltage.between(2.51, 4.19)
                        filtered_current = current[mask].copy()
                        if step_data[step_value][counter]["ampcc"] is None:
                            step_data[step_value][counter][
                                "ampcc"
                            ] = filtered_current
                        else:
                            step_data[step_value][counter]["ampcc"] = (
                                np.hstack(
                                    (
                                        step_data[step_value][counter][
                                            "ampcc"
                                        ],
                                        filtered_current,
                                    )
                                )
                            )

for step_value in [13, 14]:
    curr = step_data[step_value][1]["amp"]
    currcc = step_data[step_value][1]["ampcc"]
    dur = step_data[step_value][1]["time"]
    finAh = step_data[step_value][1]["finAh"]
    print(f"STEP {step_value}")
    print(f"\tAMP = {np.mean(curr):.10g} A +/- {np.std(curr):.5g} A ")
    print(f"\tAMP CC = {np.mean(currcc):.10g} A +/- {np.std(currcc):.5g} A ")
    print(f"\tTime = {np.mean(dur):.10g} min +/- {np.std(dur):.5g} min ")
    print(f"\tfinAh = {np.mean(finAh):.10g} Ah +/- {np.std(finAh):.5g} Ah ")


for counter in range(1, 8):
    print(f"Pulse {counter}")
    for step_value in [16, 18, 19, 21, 23]:
        print(f"\tSTEP {step_value}")
        curr = step_data[step_value][counter]["amp"]
        currcc = step_data[step_value][counter]["ampcc"]
        dur = step_data[step_value][counter]["time"]
        finAh = step_data[step_value][counter]["finAh"]
        print(f"\t\tAMP = {np.mean(curr):.10g} A +/- {np.std(curr):.5g} A ")
        print(
            f"\t\tAMP CC = {np.mean(currcc):.10g} A +/- {np.std(currcc):.5g} A "
        )
        print(f"\t\tTime = {np.mean(dur):.10g} min +/- {np.std(dur):.5g} min ")
        print(
            f"\t\tfinAh = {np.mean(finAh):.10g} Ah +/- {np.std(finAh):.5g} Ah "
        )
