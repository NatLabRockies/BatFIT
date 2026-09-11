from file_management import *
from prettyPlot.plotting import *
from read import *

from batfit import logger

# logger.setLevel("DEBUG")
logger.setLevel("INFO")


fig = plt.figure()
file_names = get_all_protocol_all_rpt_data_file(data_type="diffcap")
for protocol in file_names:
    for rpt_id in file_names[protocol]:
        for cell_in in file_names[protocol][rpt_id]:
            filename = file_names[protocol][rpt_id][cell_in]
            cycle_df = read_single_csv(filename, data_type="diffcap")
            plt.plot(
                get_elapsed_test_time(cycle_df),
                get_current(cycle_df),
                color="k",
                linewidth=1,
            )
pretty_labels("t [min]", "I [A]", 16, fontname="Times", grid=False)
# plt.show()

fig = plt.figure()
file_names = get_all_protocol_all_rpt_data_file(data_type="diffcap")
for protocol in file_names:
    for rpt_id in file_names[protocol]:
        for cell_id in file_names[protocol][rpt_id]:
            filename = file_names[protocol][rpt_id][cell_id]
            cycle_df = read_single_csv(filename, data_type="diffcap")
            A = get_voltage(cycle_df)
            if A.shape[0] > 15000:
                breakpoint()
                continue
            plt.plot(
                get_elapsed_test_time(cycle_df),
                get_voltage(cycle_df),
                color="k",
                linewidth=1,
            )
pretty_labels("t [min]", r"$\phi$ [V]", 16, fontname="Times", grid=False)


plt.show()
