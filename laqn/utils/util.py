import os
import dill as pickle
from typing import Any

def save_object(object:Any, object_name:str, path:str) -> None:
    """
    Saves an object using dill (pickle).

    Args:
        object: Object to save.
        object_name: Filename for the object.
        path: Path to save to.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    
    filename = f"{path}/{object_name}.sav"
    pickle.dump(object, open(filename, "wb"))

def load_object(object_name:str, path:str) -> Any:
    """
    Loads an object using dill (pickle).

    Args:
        object_name: Filename for the desired object.
        path: Path where object is saved.

    Returns:
        The object which matches the given filename.
    """
    filename = f"{path}/{object_name}.sav"
    return pickle.load(open(filename, 'rb'))

def load_trial(trial_path:str) -> dict:
    """
    Loads an experiment trial stored in the given folder using dill (pickle).

    An experiment trial is defined by (potentially multiple groups/iterations of) an ST-SVGP model, the pandas DataFrame used to train it,
    the scalers used for each feature, and the ordering of the most informative sites determined by Krause's algorithm. Each trial
    iteration must be numbered, starting at 1. For example, a folder containing an experiment trial of 1 iteration will have files
    "model_1.sav", "df_1.sav", "scalers_1.sav", and "krause_df_1.sav".

    Args:
        trial_path: Folder containing the experiment trial.
    
    Returns:
        A dict mapping each trial iteration to its model, df, scalers, and krause_df.
    """
    def is_trial_file(f:str):
        """
        Checks if a file belongs to a trial.

        Args:
            f: The filename.

        Returns:
            True if the file belongs to a trial.
        """
        cond1 = f.startswith("model")
        cond2 = f.startswith("df")
        cond3 = f.startswith("scalers")
        cond4 = f.startswith("krause_df")
        return cond1 or cond2 or cond3 or cond4
    
    all_files = os.listdir(trial_path)
    trial_files = [f for f in all_files if is_trial_file(f)]
    num_iters = len(trial_files)//4
    trial = dict()
    for it in range(1, num_iters+1):
        trial[it] = {
        "model": load_object(f"model_{it}", trial_path),
        "df": load_object(f"df_{it}", trial_path),
        "scalers": load_object(f"scalers_{it}", trial_path),
        "krause_df": load_object(f"krause_df_{it}", trial_path)
        }
    return trial

def load_experiment(experiment_path):
    """
    Loads an experiment stored in the given folder using dill (pickle).

    An experiment is defined by a trial(s). Each trial must be in its own folder within the experiment folder.

    Args:
        experiment_path: Folder containing the experiment trial(s).
    
    Returns:
        A dict mapping each trial name to its trial details (see load_trial above).
    """
    trial_names = os.listdir(experiment_path)
    return {trial_name: load_trial(f"{experiment_path}/{trial_name}") for trial_name in trial_names}