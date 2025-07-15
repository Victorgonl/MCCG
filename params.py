import argparse
import yaml


def set_params():
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args, unknown = parser.parse_known_args()

    with open(args.config, "r") as setting:
        args_dict = yaml.load(setting, Loader=yaml.FullLoader)

    for key, value in args_dict.items():
        parser.add_argument(f"--{key}", default=value)

    args = parser.parse_args()
    excluded_params = [
        "cuda",
        "gpu",
        "seed",
        "lr",
        "epochs",
        "dataset",
        "save_path",
        "predict_result",
        "log_dir",
        "mode",
        "layer_shape",
        "dim_proj_multiview",
        "dim_proj_cluster",
        "ground_truth_file",
        "refine",
    ]
    filtered_args_dict = {
        k: v for k, v in args_dict.items() if k not in excluded_params
    }

    return filtered_args_dict, args
