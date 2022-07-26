from datetime import datetime
import time

from laqn.modeling import model_funcs
from laqn.optimization import krause_alg, placement_sets
from laqn.utils import util


if __name__ == "__main__":
    """
    Experiment 1:
        Assess Krause's algorithm's ability to increase the mutual information of a sensor network
        by selecting novel sensor placements.
    """
    # Load raw data for the experiment
    pollutant = "NO2"
    time_step = "D"
    time_range = (datetime(2011, 1, 1), datetime(2011, 12, 31))
    cutoff = 0.8
    raw_df = model_funcs.load_data(pollutant, time_step=time_step, time_range=time_range, cutoff=cutoff)

    # Set up experiment parameters
    exp_name = "exp_1"
    results_folder = f"results/{exp_name}"
    num_trials = 10

    print(f"[Experiment] Starting experiment {exp_name}...")
    exp_t0 = time.time()
    # In each trial, randomly generate train/test/val datasets and train a model on these sets.
    # Then, select the most informative sensors and train a new model using these sensors as the 
    # train set, keeping the val set the same
    for trial_num in range(1, num_trials+1):
        print(f"[Experiment] Starting experiment trial {trial_num}...")
        trial_t0 = time.time()
        trial_name = f"{exp_name}_{trial_num}"
        results_path = f"{results_folder}/{trial_name}"

        scaled_features = ["latitude", "longitude"]
        x_features = ["t", "scaled_latitude", "scaled_longitude"]
        y_features = ["no2"]

        # 5 val/ 50 train/20 test
        num_val_sites = 5
        num_train_sites = 50
        iters = 2
        t_max = raw_df["t"].max()
        val_sites, train_sites = None, None
        first_iter = True
        for it in range(1, iters+1):
            print(f"Running iteration {it} of {trial_name}...")

            # Get train/test/val sites
            if first_iter:
                sites = model_funcs.get_data_split_sites(raw_df, num_val_sites=num_val_sites, num_train_sites=num_train_sites)
                first_iter = False
            else:
                sites = model_funcs.get_data_split_sites(raw_df, val_sites=val_sites, train_sites=train_sites)
            
            # Train model
            split_df = model_funcs.split_data_on_sites(raw_df, sites)
            df, scalers = model_funcs.scale_data(split_df, scaled_features)
            
            gp = model_funcs.train_model(df, x_features, y_features, sparse=True, num_z_space=30)

            # Run Krause's alg
            S_df, U_df = placement_sets.generate_placement_sets(df, scalers, n=1000, plot=False)
            k = S_df.shape[0]

            krause_df, cache = krause_alg.krause_alg(gp, k, S_df, U_df, t_max)

            train_sites = set(krause_df.loc[krause_df["order"] <= num_train_sites]["code"].values)
            val_sites = set(df.loc[df["dataset"] == "val"]["code"].values)

            # Save experiment trial iteration
            util.save_object(gp, f"model_{it}", results_path)
            util.save_object(df, f"df_{it}", results_path)
            util.save_object(scalers, f"scalers_{it}", results_path)
            util.save_object(krause_df, f"krause_df_{it}", results_path)
        
        trial_t1 = time.time()
        trial_t = trial_t1-trial_t0
        print(f"[Experiment] Time to complete trial {trial_name}: {trial_t:.2f} s")

    exp_results = util.load_experiment(results_folder)

    exp_t1 = time.time()
    exp_t = exp_t1-exp_t0
    print(f"[Experiment] Time to complete experiment {exp_name}: {exp_t:.2f} s")