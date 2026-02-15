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

    # Optional: define a custom order by sections if desired
    # Otherwise, it will dump keys in arbitrary order
    with open(yaml_path, "w") as f:
        f.write("# DATA\n")
        yaml.dump(
            {
                "constrain_output": d["constrain_output"],
                "cyc_mode": d["cyc_mode"],
                "data_path": d["data_path"],
                "data_val_path": d["data_val_path"],
                "n_param_pred": d["n_param_pred"],
                "n_points": d["n_points"],
                "scaler_path": d["scaler_path"],
                "sim_config": d["sim_config"],
            },
            f,
            default_flow_style=False,
        )
        f.write("\n# ARCH\n")
        yaml.dump(
            {
                "fc_units": d["fc_units"],
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
    abs_data_path = sys.argv[3]
    abs_sim_config_path = sys.argv[4]

    # Load yaml file with relative path
    with open(input_yaml_file, "r") as file:
        input_data = yaml.safe_load(file)

    # Replace data with absolute path
    input_data["data_path"] = os.path.join(
        abs_data_path, input_data["data_path"]
    )
    input_data["data_val_path"] = os.path.join(
        abs_data_path, input_data["data_val_path"]
    )
    input_data["scaler_path"] = os.path.join(
        abs_data_path, input_data["scaler_path"]
    )
    input_data["sim_config"] = os.path.join(
        abs_sim_config_path, input_data["sim_config"]
    )
    input_data["models_dir"] = os.path.join(
        abs_data_path, "2.train_surrogate", input_data["models_dir"]
    )

    # Write new yaml
    dict_to_yaml(d=input_data, yaml_path=output_yaml_file)
