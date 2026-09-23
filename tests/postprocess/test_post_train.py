import os
import tempfile

import matplotlib

matplotlib.use("Agg")


from batfit.postprocess.post_train import plot_loss


def test_plot_loss():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Write a dummy train loss
        loss_file = os.path.join(tmp_dir, "train_loss.csv")
        with open(loss_file, "w") as f:
            f.write("step;loss\n")
            for i in range(10):
                f.write(f"{i};{1.0 / (i + 1):.6f}\n")

        figure_folder = os.path.join(tmp_dir, "figures")
        plot_loss(loss_file, figure_folder=figure_folder, fig_name="loss.png")

        assert os.path.exists(os.path.join(figure_folder, "loss.png"))
