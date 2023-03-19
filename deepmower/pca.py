#%%

import pandas as pd

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#%%

def drop_constant_column(dataframe):
    """
    Drops constant value columns of pandas dataframe.
    """
    return dataframe.loc[:, (dataframe != dataframe.iloc[0]).any()]

#%%

def get_go_paths(log_f_name = None):
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
           'End_y', 'Perc_done', 'Frames_Since_Fuel', 'Momentum Lost']]


    run_df = drop_constant_column(run_df)

    scaler = StandardScaler()
    scaled_df=run_df.copy()
    scaled_df=pd.DataFrame(scaler.fit_transform(scaled_df), columns=scaled_df.columns)



    #define PCA model to use
    pca = PCA(n_components=4)

    #fit PCA model to data
    pca_fit = pca.fit(scaled_df)



    pcs = scaled_df.__matmul__(pca.components_.T)


    idxs = []

    for i in range(4):
        idxs.append(pcs[i].argmax())
        idxs.append(pcs[i].argmin())

    paths = paths.iloc[idxs].reset_index(drop=True)

    paths_list = paths.values.tolist()

    go_paths = []

    for path_ in paths_list:
        path = path_.strip('][').split(', ')
        path = list(map(int,path))
        path_length = len(path)
        end_idx = int(np.random.triangular(left = int(0.1*path_length),
                                       right = int(0.9*path_length),
                                       mode = int(0.7*path_length)))
        go_paths.append(path[:end_idx])

    return go_paths