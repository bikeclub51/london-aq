import bayesnewton
import numpy as np
import pandas as pd
from scipy.linalg import det
import time

def mutual_info(GP:bayesnewton.models.MarkovVariationalGP, k:int, krause_df:pd.DataFrame, U_df:pd.DataFrame, t_max:int, t_min:int=0) -> np.array:
    """
    Calculates the mutual information of the sensor network.

    Uses the equation presented here:
    https://www.math.nyu.edu/~kleeman/infolect7.pdf

    Args:
        GP: A trained ST-SVGP model.
        k: The number of sensors to calculate the mutual information for.
        krause_df: A pandas DataFrame representing the most informative sensor placements with columns ["code", "site_score", "order"]
            where "code" represents the site code, "site_score" represents the y* value for the chosen site, and "order" represents the
            ranking of the sites. Has the same form as the output of the krause_alg function.
        U_df: pandas DataFrame representing the set U of positions of interest where no sensor placements are possible.
            Must have columns ["code", "scaled_latitude", "scaled_longitude"].
        t_max: Maximum time step to average the covariance matrix over.
        t_min: Minimum time step to average the covariance matrix over.

    Returns:
        A numpy array containing the mutual information values as each of the k sensors is added to the network in the order established by the
        input krause_df.
    """
    print(f"[MI] Starting mutual information calculation...")
    MI = np.array([],dtype=float)

    # Build cache for faster compute
    cache_t0 = time.time()
    cov_mats = []
    all_df = krause_df.append(U_df)
    for t in range(t_min, t_max+1):
        all_t = np.array([[t]])
        all_R = np.tile(all_df[["scaled_latitude", "scaled_longitude"]].values, [all_t.shape[0], 1, 1])
        all_mean, all_cov = GP.likelihood_cov(X=all_t, R=all_R)
        cov_mats.append(all_cov)

    cov_mat = np.mean(cov_mats, axis=0)
    cache_t1 = time.time()
    cache_t = cache_t1-cache_t0
    print(f"[MI] Time to fill cache ({t_max-t_min+1} timesteps): {cache_t:.2f} s")

    all_sites = all_df["code"].tolist()
    site_to_ind = {site: ind for ind, site in enumerate(all_sites)}

    krause_sites = krause_df["code"].tolist()
    U_sites = U_df["code"].tolist()
    num_U_sites = len(U_sites)

    # Calculate mutual information for each added sensor
    calc_t0 = time.time()
    for i in range(1, k+1):
        chosen_sites = krause_sites[:i]
        current_sites = chosen_sites + U_sites
        
        cov_mat_XU = np.zeros((i+num_U_sites, i+num_U_sites))

        for ind_1, site_1 in enumerate(current_sites):
            for ind_2, site_2 in enumerate(current_sites):
                cov_mat_XU[ind_1, ind_2] = cov_mat[site_to_ind[site_1], site_to_ind[site_2]].item()

        XX_cov = cov_mat_XU[:i,:i]
        UU_cov = cov_mat_XU[i:, i:]
        
        det_XU = det(cov_mat_XU)
        det_XX = det(XX_cov)
        det_UU = det(UU_cov)
        
        mutual_info = 0.5*np.log((det_XX * det_UU) / (det_XU))
        MI = np.append(MI, mutual_info)
    
    calc_t1 = time.time()
    calc_t = calc_t1-calc_t0
    print(f"[MI] Time to calculate mutual information ({t_max-t_min+1} timesteps; {k} sites): {calc_t:.2f} s")
    
    return MI