#%%

import pandas as pd

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans


#%%

def drop_constant_column(dataframe):
    """
    Drops constant value columns of pandas dataframe.
    """
    return dataframe.loc[:, (dataframe != dataframe.iloc[0]).any()]

#%%

def get_go_paths(log_f_name = None, n_pcs = 6, pca_type = 0, n_sample = 12):
    if log_f_name == None:
        go_explore = False
        reward_type = 1
        lawn_num = 21
        run_id = 1

        log_dir = "PPO_logs"

        sub_dir = 'go_explore_' + str(go_explore) + '/reward_function' + str(reward_type) + "/"

        env_name = f"lawn{lawn_num}"
        log_dir = log_dir + '/' + env_name + '/' + sub_dir
        run_num = run_id
        log_f_name = log_dir + '/' + str(run_num) + ".csv"''

    run_df = pd.read_csv(log_f_name)

    paths = run_df['Path']

    run_df = run_df[['Score', 'Fuel_Score', 'Grass_Score',
           'Num_Fuel_Obtained', 'Amt_Fuel_Obtained', 'End_Fuel', 'Frames', 'End_x',
           'End_y', 'Perc_done', 'Frames_Since_Fuel', 'Fuel_Manhattan', 'Momentum Lost']]

    run_df = drop_constant_column(run_df)

    scaler = StandardScaler()
    scaled_df=run_df.copy()
    scaled_df=pd.DataFrame(scaler.fit_transform(scaled_df), columns=scaled_df.columns)

    # in case n_pcs is too big
    n_pcs = min(n_pcs, scaled_df.shape[1])

    #define PCA model to use
    pca = PCA(n_components=n_pcs)

    #fit PCA model to data
    pca.fit(scaled_df)



    pcs = scaled_df.__matmul__(pca.components_.T)

    if pca_type == 0:
        # original
        idxs = []

        for i in range(n_pcs):
            idxs.append(pcs[i].argmax())
            idxs.append(pcs[i].argmin())

    elif pca_type == 1:
        # weighing based on all pcs
        pcs_old = pcs
        weights = np.array(np.abs(pcs_old)).__matmul__(np.array(np.sqrt(pca.explained_variance_)))

        pcs['weights'] = weights
        idxs = pcs.sample(n=n_sample, replace=False, weights='weights').index

    elif pca_type == 2:
        # sample pc's based on var explained, then sample two extremes
        weights = pca.explained_variance_
        weights = weights / np.sum(weights)
        pc_idxs = np.random.choice(range(n_pcs), size=int(n_sample / 2), replace=False, p=weights)
        idxs = []
        for pc_idx in pc_idxs:
            pc = pcs[pc_idx]
            wt1 = np.exp(pc)
            wt2 = np.exp(-pc)
            idx1 = pc.sample(n=1, weights=wt1).index[0]
            idx2 = pc.sample(n=1, weights=wt2).index[0]
            idxs.append(idx1)
            idxs.append(idx2)

    elif pca_type == 3:
        X = pcs
        X.columns = X.columns.astype(str)
        kmeans = KMeans(n_clusters=n_sample).fit(pcs)

        # %%

        idxs = []
        for i in range(n_sample):
            d = kmeans.transform(X)[:, i]
            idxs.append(d.argmin())

    elif pca_type == 4:
        X = pcs
        X.columns = X.columns.astype(str)
        kmeans = KMeans(n_clusters=int(n_sample / 2)).fit(pcs)


        idxs = []
        kmx = kmeans.transform(X)
        for i in range(int(n_sample / 2)):
            d_all = kmx.sum(axis=1)
            d_nearest = kmx[:, i]
            d_all_but_nearest = d_all - d_nearest
            idxs.append(d_all_but_nearest.argmin())

    # get paths based on indices
    paths = paths.iloc[idxs].reset_index(drop=True)

    paths_list = paths.values.tolist()

    go_paths = []

    for path_ in paths_list:
        path = path_.strip('][').split(', ')
        path = list(map(int,path))
        path_length = len(path)
        try:
            # end_idx = int(np.random.triangular(left = int(0.1*path_length),
            #                                right = int(0.9*path_length),
            #                                mode = int(0.7*path_length)))
            end_idx = int(np.random.uniform(low = int(0.1*path_length),
                                           high = int(0.9*path_length)))
            go_paths.append(path[:end_idx])
        except:
            print(f"-- warning -- skipping path.  path length = {len(path)}.  path:")
            print(path)
            print(f"-------------")

    return go_paths