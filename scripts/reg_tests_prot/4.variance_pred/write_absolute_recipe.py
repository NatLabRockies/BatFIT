"""
Convert relative paths in recipe_gen_dataset.yml and recipe_var_pred.yml to
absolute paths and write the resolved files to training_recipes/.

Usage:
    python write_absolute_recipe.py <abs_reg_path> <abs_batfit_exp_path>

where:
    abs_reg_path       — absolute path to scripts/reg_tests_prot/
    abs_batfit_exp_path — absolute path to batfit/default_exps/
"""

import os
import sys

import yaml


def _dump_yaml(d: dict, path: str, sections: list[list[str]]) -> None:
    """Write a dictionary to a YAML file with section comments."""

    def represent_list_inline(dumper, data):
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq", data, flow_style=True
        )

    yaml.add_representer(list, represent_list_inline)

    with open(path, "w") as f:
        for keys in sections:
            yaml.dump(
                {k: d[k] for k in keys if k in d},
                f,
                default_flow_style=False,
            )
            f.write("\n")


def write_gen_dataset_recipe(
    input_yaml: str, output_yaml: str, abs_reg_path: str, abs_exp_path: str
) -> None:
    """Resolve paths in recipe_gen_dataset.yml."""
    with open(input_yaml, "r") as f:
        d = yaml.safe_load(f)

    d["npe_models_dir"] = os.path.join(
        abs_reg_path, "2.npe_cnn_prot", d["npe_models_dir"]
    )
    d["data_path"] = os.path.join(abs_reg_path, "1.gen_data", d["data_path"])
    d["scaler_path"] = os.path.join(
        abs_reg_path, "1.gen_data", d["scaler_path"]
    )
    d["scaler_P_path"] = os.path.join(
        abs_reg_path, "1.gen_data", d["scaler_P_path"]
    )
    d["var_pred_save_path"] = os.path.join(
        abs_reg_path, "4.variance_pred", d["var_pred_save_path"]
    )

    _dump_yaml(
        d,
        output_yaml,
        sections=[
            ["npe_models_dir"],
            [
                "cyc_mode",
                "data_path",
                "n_points",
                "n_prot_params",
                "noise_factor",
                "scaler_path",
                "scaler_P_path",
                "target_mode",
            ],
            ["gen_batch_size", "n_noise", "use_true_y", "var_pred_save_path"],
        ],
    )


def write_var_pred_recipe(
    input_yaml: str, output_yaml: str, abs_reg_path: str, abs_exp_path: str
) -> None:
    """Resolve paths in recipe_var_pred.yml."""
    with open(input_yaml, "r") as f:
        d = yaml.safe_load(f)

    d["sim_config"] = os.path.join(abs_exp_path, d["sim_config"])
    d["var_pred_save_path"] = os.path.join(
        abs_reg_path, "4.variance_pred", d["var_pred_save_path"]
    )
    d["models_dir"] = os.path.join(
        abs_reg_path, "4.variance_pred", d["models_dir"]
    )

    _dump_yaml(
        d,
        output_yaml,
        sections=[
            [
                "n_param_pred",
                "n_prot_params",
                "sim_config",
                "var_pred_save_path",
            ],
            ["hidden_list"],
            ["batch_size", "epochs", "lr", "models_dir"],
        ],
    )


if __name__ == "__main__":
    abs_reg_path = sys.argv[1]
    abs_exp_path = sys.argv[2]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    recipes_dir = os.path.join(script_dir, "training_recipes")

    write_gen_dataset_recipe(
        input_yaml=os.path.join(recipes_dir, "recipe_gen_dataset.yml"),
        output_yaml=os.path.join(recipes_dir, "recipe_gen_dataset_abs.yml"),
        abs_reg_path=abs_reg_path,
        abs_exp_path=abs_exp_path,
    )
    write_var_pred_recipe(
        input_yaml=os.path.join(recipes_dir, "recipe_var_pred.yml"),
        output_yaml=os.path.join(recipes_dir, "recipe_var_pred_abs.yml"),
        abs_reg_path=abs_reg_path,
        abs_exp_path=abs_exp_path,
    )
    print("Absolute recipes written to training_recipes/")
