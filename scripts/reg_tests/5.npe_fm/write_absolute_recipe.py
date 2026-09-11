import os
import sys

import yaml


def dict_to_yaml(d: dict, yaml_path: str):
    """
    Save a Python dictionary to a YAML file.

    Args:
        d (dict): Dictionary to save.
        yaml_path (str): Path to output YAML file.
    """

    # force lists to be represented inline
    def represent_list_inline(dumper, data):
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq", data, flow_style=True
        )

    yaml.add_representer(list, represent_list_inline)

    with open(yaml_path, "w") as f:
        f.write("# DATA\n")
        yaml.dump(
            {
                "cyc_mode": d["cyc_mode"],
                "data_path": d["data_path"],
                "data_val_path": d["data_val_path"],
                "n_param_pred": d["n_param_pred"],
                "n_points": d["n_points"],
                "noise_factor": d["noise_factor"],
                "sim_config": d["sim_config"],
                "target_mode": d["target_mode"],
            },
            f,
            default_flow_style=False,
        )
        f.write("\n# ARCH - CNN encoder\n")
        yaml.dump(
            {
                "num_channels": d["num_channels"],
                "num_convs": d["num_convs"],
                "num_fc_hidden": d["num_fc_hidden"],
                "num_fc_units": d["num_fc_units"],
            },
            f,
            default_flow_style=False,
        )
        f.write("\n# ARCH - velocity field MLP\n")
        yaml.dump(
            {
                "num_vf_hidden": d["num_vf_hidden"],
                "num_vf_units": d["num_vf_units"],
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
        f.write("\n# SURROGATE\n")
        yaml.dump(
            {"surrogate_model_recipe": d["surrogate_model_recipe"]},
            f,
            default_flow_style=False,
        )


if __name__ == "__main__":
    input_yaml_file = sys.argv[1]
    output_yaml_file = sys.argv[2]
    abs_reg_path = sys.argv[3]
    abs_sim_config_path = sys.argv[4]

    # Load yaml file with relative path
    with open(input_yaml_file, "r") as file:
        input_data = yaml.safe_load(file)

    # Replace data with absolute path
    input_data["data_path"] = os.path.join(
        abs_reg_path, "1.gen_data", input_data["data_path"]
    )
    input_data["data_val_path"] = os.path.join(
        abs_reg_path, "1.gen_data", input_data["data_val_path"]
    )
    input_data["sim_config"] = os.path.join(
        abs_sim_config_path, input_data["sim_config"]
    )
    input_data["models_dir"] = os.path.join(
        abs_reg_path, "5.npe_fm", input_data["models_dir"]
    )
    input_data["surrogate_model_recipe"] = os.path.join(
        abs_reg_path, "2.train_surrogate", input_data["surrogate_model_recipe"]
    )

    # Write new yaml
    dict_to_yaml(d=input_data, yaml_path=output_yaml_file)
