from datetime import datetime
import time

from laqn.modeling import model_funcs
from laqn.optimization import krause_alg, placement_sets
from laqn.utils import util

if __name__ == "__main__":
    """
    Experiment 2:
        Optimize the sensor locations of the LAQN by allowing Krause's algorithm to select sensor placements
        at locations all across London (not just locations where we have data).
    """
    num_iters = 1
    for exp_num in range(1, num_iters+1):
        print(f"[Experiment] Starting experiment {exp_num}...")
        exp_t0 = time.time()
        experiment_name = f"rec_exp_{exp_num}"

        # Load raw data for the experiment
        pollutant = "NO2"
        time_step = "D"
        time_range = (datetime(2011, 1, 1), datetime(2011, 12, 31))
        cutoff = 0.8

        raw_df = model_funcs.load_data(pollutant, time_step=time_step, time_range=time_range, cutoff=cutoff)

        # Train model on randomly split test/train (80%/20%) sites 
        scaled_features = ["latitude", "longitude"]
        x_features = ["t", "scaled_latitude", "scaled_longitude"]
        y_features = ["no2"]

        t_max = raw_df["t"].max()

        split_df = model_funcs.split_data_randomly(raw_df, test_size=0.2)
        df, scalers = model_funcs.scale_data(split_df, scaled_features)

        gp = model_funcs.train_model(df, x_features, y_features, sparse=True, num_z_space=30)

        S_df, U_df = placement_sets.generate_placement_sets(df, scalers, n=1000, plot=False)
        num_sites = S_df.shape[0]
        print(num_sites)
        num_U_sites = U_df.shape[0]
        print(num_U_sites)

        # Choose best existing 50 sites
        num_kept = 50
        krause_df, cache = krause_alg.krause_alg(gp, num_sites, S_df, U_df, t_max)
        print(S_df.shape)
        print(U_df.shape)
        print(krause_df.shape)
        print(krause_df.head())
        kept_krause_df = krause_df.loc[krause_df["order"] <= num_kept]

        # Choose best new sites for remaining sensors
        new_S_df = kept_krause_df.append(U_df)
        print(new_S_df.shape)
        print(new_S_df.head())

        new_krause_df, cache = krause_alg.krause_alg(gp, num_sites-num_kept, new_S_df, None, t_max, A_df=kept_krause_df)

        print(new_krause_df.shape)
        print(new_krause_df.tail())

        # Save experiment trial iteration
        util.save_object(gp, f"model_{exp_num}", experiment_name)
        util.save_object(df, f"df_{exp_num}", experiment_name)
        util.save_object(scalers, f"scalers_{exp_num}", experiment_name)
        util.save_object(krause_df, f"krause_df_{exp_num}", experiment_name)
        util.save_object(new_krause_df, f"new_krause_df_{exp_num}", experiment_name)

        exp_t1 = time.time()
        exp_t = exp_t1-exp_t0
        print(f"[Experiment] Time to complete experiment: {exp_t:.2f} s")