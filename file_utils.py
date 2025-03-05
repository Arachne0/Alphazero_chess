import os
import wandb


def initialize_wandb(args, n_playout=None):
    common_config = {
        "entity": "hails",
        "project": "gym_chess",
        "config": args.__dict__
    }

    run_name = f"Chess-MCTS{n_playout}"
    wandb.init(name=run_name, **common_config)


def create_models(n_playout, i=None):
    """
    Generate training and evaluation model file paths dynamically based on the rl_model and parameters.
    """
    base_paths = {
        "Training": "Training",
        "Eval": "Eval"
    }

    # Construct the specific path part for the model
    specific_path = f"nmcts{n_playout}"
    filename = f"train_{i + 1:03d}.pth"

    # Generate full paths
    model_file = f"{base_paths['Training']}/{specific_path}/{filename}"
    eval_model_file = f"{base_paths['Eval']}/{specific_path}/{filename}"

    return model_file, eval_model_file


def get_existing_files(n_playout):
    """
    Retrieve a list of existing file indices based on the model type and parameters.
    """
    base_path = "Training"
    path = f"{base_path}/nmcts{n_playout}"

    # Fetch files and extract indices
    return [
        int(file.split('_')[-1].split('.')[0])
        for file in os.listdir(path)
        if file.startswith('train_')
    ]
