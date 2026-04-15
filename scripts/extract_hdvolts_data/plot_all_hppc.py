from file_management import *
from prettyPlot.plotting import *
from read import *

from batfit import logger

# logger.setLevel("DEBUG")
logger.setLevel("INFO")


# for protocol in file_names:
#    for rpt_id in file_names[protocol]:
#        for cell_in in file_names[protocol][rpt_id]:
#            fig = plt.figure()
#            filename = file_names[protocol][rpt_id][cell_in]
#            cycle_df = read_single_csv(filename)
#
#            plt.plot(get_elapsed_test_time(cycle_df), get_voltage(cycle_df), color='k', linewidth=1)
#            pretty_labels("t [min]", r"$\phi$ [V]", 16, fontname="Times", grid=False)
#            if len(cycle_df)<20000:
#                print(filename)
#                plt.show()
#            else:
#                plt.close()


#fig = plt.figure()
#file_names = get_all_protocol_all_rpt_data_file(data_type="hppc")
#for protocol in file_names:
#    for rpt_id in file_names[protocol]:
#        for cell_in in file_names[protocol][rpt_id]:
#            filename = file_names[protocol][rpt_id][cell_in]
#            cycle_df = read_single_csv(filename, data_type="hppc")
#            plt.plot(
#                get_elapsed_test_time(cycle_df),
#                get_current(cycle_df),
#                color="k",
#                linewidth=1,
#            )
#pretty_labels("t [min]", "I [A]", 16, fontname="Times", grid=False)
## plt.show()

fig = plt.figure()
file_names = get_all_protocol_all_rpt_data_file(data_type="posthppc")
for protocol in file_names:
    for rpt_id in file_names[protocol]:
        for cell_in in file_names[protocol][rpt_id]:
            #print(f"{protocols} RPT{rpt_id} Cell {cell_in}")
            filename = file_names[protocol][rpt_id][cell_in]
            cycle_df = read_single_csv(filename, data_type="posthppc")
            A = get_voltage(cycle_df).to_numpy()
            B=get_elapsed_test_time(cycle_df).to_numpy()
            ind=np.argwhere(A>3.9)
            if B[ind[0]]<100:
                print(f"\n\n ERROR for {protocol} RPT {rpt_id}  Cell {cell_in}\n\n")
                breakpoint()
            plt.plot(
                get_elapsed_test_time(cycle_df),
                get_voltage(cycle_df),
                color="k",
                linewidth=1,
            )
pretty_labels("t [min]", fr"$\phi$ [V]", 16, fontname="Times", grid=False)


plt.show()
