import bayesnewton
import numpy as np
import pandas as pd
import time

def krause_alg(GP:bayesnewton.models.MarkovVariationalGP, k:int, S_df:pd.DataFrame, U_df:pd.DataFrame, t_max:int, t_min:int=0, A_df:pd.DataFrame=None) -> pd.DataFrame:
    """
    Greedily selects the k most informative sensor placements from set S across the area covered by sets S and U.

    Follows the pseudo-code for Algorithm 1 in the original Krause paper:
    https://jmlr.org/papers/volume9/krause08a/krause08a.pdf

    Args:
        GP: A trained ST-SVGP model.
        k: The number of most informative sensor placements to select.
        S_df: pandas DataFrame representing the set S of possible locations where we can place sensors.
            Must have columns ["code", "scaled_latitude", "scaled_longitude"].
        U_df: pandas DataFrame representing the set U of positions of interest where no sensor placements are possible.
            Must have columns ["code", "scaled_latitude", "scaled_longitude"].
        t_max: Maximum time step to average the covariance matrix over.
        t_min: Minimum time step to average the covariance matrix over.
        A_df: pandas DataFrame with the same form as the output of this function. If provided, represents the top n sensor
            placements where n is the number of rows in the DataFrame. Therefore, this function will then calculate the
            next k most informative sensor placements.

    Returns:
        A pandas DataFrame representing the top k most informative sensor placements with columns ["code", "site_score", "order"]
        where "code" represents the site code, "site_score" represents the y* value for the chosen site, and "order" represents the
        ranking of the sites. Also returns the generated covariance matrix, which can be saved for faster plot times.
    """
    print(f"[Krause] Starting Krause algorithm...")
    cache_t0 = time.time()
    # Build cache for faster compute
    if U_df is not None:
        all_df = S_df.append(U_df)
    else:
        all_df = S_df.copy()
    cov_mats = []
    for t in range(t_min, t_max+1):
        all_t = np.array([[t]])
        all_R = np.tile(all_df[["scaled_latitude", "scaled_longitude"]].values, [all_t.shape[0], 1, 1])
        all_mean, all_cov = GP.likelihood_cov(X=all_t, R=all_R)
        cov_mats.append(all_cov)

    cov_mat = np.mean(cov_mats, axis=0)
    cache_t1 = time.time()
    cache_t = cache_t1-cache_t0
    print(f"[Krause] Time to fill cache ({t_max-t_min+1} timesteps): {cache_t:.2f} s")

    # Check if we've already calculated some of the most informative sites
    if A_df is not None:
        A = A_df["code"].tolist()
        A_scores = A_df["site_score"].tolist()
    else:
        A = list()
        A_scores = list()

    num_A = len(A)

    # Get list of site sets S and U
    sites = all_df["code"].tolist()
    site_to_ind = {site: ind for ind, site in enumerate(sites)}

    S_sites = S_df["code"].tolist()
    if U_df is not None:
        U_sites = U_df["code"].tolist()
    else:
        U_sites = []

    # Run krause algorithm
    calc_t0 = time.time()
    for i in range(1, k+1):
        iter_t0 = time.time()
        y_star_code = None
        delta_y_star = None

        S_diff_A = [site for site in S_sites if site not in A]
        for y in S_diff_A:
            y_ind = site_to_ind[y]
            y_var = cov_mat[y_ind, y_ind].item()
            if len(A) == 0:
                numerator = y_var
            else:
                yA = [y] + A
                yA_cov = np.zeros((len(yA), len(yA)))
                
                for yA_ind_1, site_1 in enumerate(yA):
                    for yA_ind_2, site_2 in enumerate(yA):
                        yA_cov[yA_ind_1, yA_ind_2] = cov_mat[site_to_ind[site_1], site_to_ind[site_2]].item()
                
                yA_cov_row = yA_cov[0, 1:]
                AA_cov_mat = yA_cov[1:, 1:]
                numerator = y_var - yA_cov_row@np.linalg.inv(AA_cov_mat)@yA_cov_row.T

            A_y = A + [y]
            S_diff_A_y = [site for site in S_sites if site not in A_y]
            yA_bar = [y] + S_diff_A_y + U_sites
            
            yA_bar_cov = np.zeros((len(yA_bar), len(yA_bar)))
            for yA_bar_ind_1, site_1 in enumerate(yA_bar):
                for yA_bar_ind_2, site_2 in enumerate(yA_bar):
                    yA_bar_cov[yA_bar_ind_1, yA_bar_ind_2] = cov_mat[site_to_ind[site_1], site_to_ind[site_2]].item()

            yA_bar_cov_row = yA_bar_cov[0, 1:]
            AA_bar_cov_mat = yA_bar_cov[1:, 1:]

            denominator = y_var - yA_bar_cov_row@np.linalg.inv(AA_bar_cov_mat)@yA_bar_cov_row.T
            delta_y = numerator/denominator
                
            if not y_star_code:
                y_star_code = y
                delta_y_star = delta_y
            elif delta_y > delta_y_star:
                y_star_code = y
                delta_y_star = delta_y

        A.append(y_star_code)
        A_scores.append(delta_y_star)

        iter_t1 = time.time()
        iter_t = iter_t1-iter_t0
        print(f"[Krause] Iter {i} time: {iter_t:.2f} s")
    calc_t1 = time.time()
    calc_t = calc_t1-calc_t0
    print(f"[Krause] Time to execute Krause algorithm ({t_max-t_min+1} timesteps; {k} sites): {calc_t:.2f} s")
    
    new_A_df = pd.DataFrame({"code": A, "site_score": A_scores, "order": list(range(1, num_A+k+1))})
    new_A_df = new_A_df.merge(S_df, on="code")
    
    return new_A_df.copy(), cov_mat

