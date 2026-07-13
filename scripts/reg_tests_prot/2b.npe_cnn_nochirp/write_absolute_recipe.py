import os
import sys

import yaml


def dict_to_yaml(d: dict, yaml_path: str):
    """Save a Python dictionary to a YAML file."""

    def represent_list_inline(dumper, data):
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq", data, flow_style=True
        )

    yaml.add_representer(list, represent_list_inline)

    with open(yaml_path, "w") as f:
        f.write("# DATA\n")
        yaml.dump(
            {
                k: d[k]
                for k in [
                    "cyc_mode",
                    "data_path",
                    "n_param_pred",
                    "n_points",
                    "noise_factor",
                    "sim_config",
                    "target_mode",
                ]
            },
            f,
            default_flow_style=False,
        )
        f.write("\n# ARCH\n")
        yaml.dump(
            {
                k: d[k]
                for k in [
                    "num_channels",
                    "num_convs",
                    "num_fc_hidden",
                    "num_fc_units",
                    "num_fc_gamma_mu_hidden",
                    "num_fc_gamma_mu_units",
                ]
            },
            f,
            default_flow_style=False,
        )
        f.write("\n# TRAIN\n")
        yaml.dump(
            {k: d[k] for k in ["epochs", "batch_size", "lr", "models_dir"]},
            f,
            default_flow_style=False,
        )


if __name__ == "__main__":
    input_yaml_file = sys.argv[1]
    output_yaml_file = sys.argv[2]
    abs_reg_path = sys.argv[3]
    abs_sim_config_path = sys.argv[4]

    with open(input_yaml_file, "r") as f:
        input_data = yaml.safe_load(f)

    input_data["data_path"] = os.path.join(
        abs_reg_path, "1b.gen_data_nochirp", input_data["data_path"]
    )
    input_data["sim_config"] = os.path.join(
        abs_sim_config_path, input_data["sim_config"]
    )
    input_data["models_dir"] = os.path.join(
        abs_reg_path, "2b.npe_cnn_nochirp", input_data["models_dir"]
    )

    dict_to_yaml(d=input_data, yaml_path=output_yaml_file)
