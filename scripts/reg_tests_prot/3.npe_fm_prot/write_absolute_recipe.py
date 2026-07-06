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
                    "n_prot_params",
                    "n_points",
                    "noise_factor",
                    "scaler_path",
                    "scaler_P_path",
                    "sim_config",
                    "target_mode",
                ]
            },
            f,
            default_flow_style=False,
        )
        f.write("\n# ARCH - CNN encoder\n")
        yaml.dump(
            {
                k: d[k]
                for k in [
                    "num_channels",
                    "num_convs",
                    "num_fc_hidden",
                    "num_fc_units",
                    "num_fc_prot_hidden",
                    "num_fc_prot_units",
                ]
            },
            f,
            default_flow_style=False,
        )
        f.write("\n# ARCH - velocity field MLP\n")
        yaml.dump(
            {
                k: d[k]
                for k in [
                    "num_vf_hidden",
                    "num_vf_units",
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
        f.write("\n# FM SPECIFIC\n")
        yaml.dump(
            {"use_prior_matching": d["use_prior_matching"]},
            f,
            default_flow_style=False,
        )
        f.write("\n# TEST\n")
        yaml.dump(
            {
                "n_samples": d["n_samples"],
                "n_ode_steps": d["n_ode_steps"],
            },
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
        abs_reg_path, "1.gen_data", input_data["data_path"]
    )
    input_data["scaler_path"] = os.path.join(
        abs_reg_path, "1.gen_data", input_data["scaler_path"]
    )
    input_data["scaler_P_path"] = os.path.join(
        abs_reg_path, "1.gen_data", input_data["scaler_P_path"]
    )
    input_data["sim_config"] = os.path.join(
        abs_sim_config_path, input_data["sim_config"]
    )
    input_data["models_dir"] = os.path.join(
        abs_reg_path, "3.npe_fm_prot", input_data["models_dir"]
    )

    dict_to_yaml(d=input_data, yaml_path=output_yaml_file)
