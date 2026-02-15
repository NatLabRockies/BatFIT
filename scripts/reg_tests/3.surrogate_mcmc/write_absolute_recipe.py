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
        f.write("# Uncertainty\n")
        yaml.dump(
            {
                "min_sigma" : d["min_sigma"],
                "max_sigma" : d["max_sigma"],
                "calibrate_sigma" : d["calibrate_sigma"],
            },
            f,
            default_flow_style=False,
        )
        f.write("\n# MCMC\n")
        yaml.dump(
            {
                "mcmc_method": d["mcmc_method"],
                "num_chains": d["num_chains"],
                "num_warmup": d["num_warmup"],
                "num_samples": d["num_samples"],
            },
            f,
            default_flow_style=False,
        )
        f.write("\n# DATA processing\n")
        yaml.dump(
            {
                "noise_factor": d["noise_factor"],
                "data_path_discharge": d["data_path_discharge"],
                "step_size": d["step_size"],
                "cyc_mode": d["cyc_mode"],
                "target_mode": d["target_mode"],
            },
            f,
            default_flow_style=False,
        )
        f.write("\n# Surrogate Models\n")
        yaml.dump(
            {
                "model_discharge_recipe": d["model_discharge_recipe"],
            },
            f,
            default_flow_style=False,
        )
        f.write("\n# DATA\n")
        yaml.dump(
            {
                "n_param_pred": d["n_param_pred"],
                "n_points": d["n_points"],
            },
            f,
            default_flow_style=False,
        )
        f.write("\n# models\n")
        yaml.dump(
            {
                "models_dir": d["models_dir"],
            },
            f,
            default_flow_style=False,
        )



if __name__ == "__main__":
    input_yaml_file = sys.argv[1]
    output_yaml_file = sys.argv[2]
    abs_reg_path = sys.argv[3]

    # Load yaml file with relative path
    with open(input_yaml_file, 'r') as file:
        input_data = yaml.safe_load(file)
   

    # Replace data with absolute path
    input_data["data_path_discharge"] = os.path.join(abs_reg_path, "1.gen_data", input_data["data_path_discharge"])
    input_data["model_discharge_recipe"] = os.path.join(abs_reg_path, "2.train_surrogate", input_data["model_discharge_recipe"])
    input_data["models_dir"] = os.path.join(abs_reg_path, "3.surrogate_mcmc", input_data["models_dir"])
    
    # Write new yaml
    dict_to_yaml(d=input_data, yaml_path=output_yaml_file)

