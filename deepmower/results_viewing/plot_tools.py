
import matplotlib.pyplot as plt
import pandas as pd



#%%

def plot_loss(
        lawn_num,
        run_id,
        reward_function,
        go_explore = 'False',
        save = False,
        ax = None,
        legend = False
        ):



    csv_path = f"../PPO_logs/lawn{lawn_num}/go_explore_{go_explore}/reward_function{reward_function}/{run_id}_loss_plots.csv"

    #%%

    loss_data = pd.read_csv(csv_path, index_col = 'iter')
    loss_data['total_loss_10'] = loss_data['total_loss'].rolling(10).mean()
    loss_data['value_loss_10'] = loss_data['value_loss'].rolling(10).mean()
    loss_data['action_loss_10'] = loss_data['action_loss'].rolling(10).mean()
    loss_data['entropy_loss_10'] = loss_data['entropy_loss'].rolling(10).mean()
    #%%

    loss_data.dropna(inplace = True)
    #%%

    if ax is None:
        fig, ax3 = plt.subplots()
    else:
        ax3 = ax


    ax3.plot(loss_data.index, loss_data['total_loss_10'], label='Total Loss')
    ax3.plot(loss_data.index, loss_data['entropy_loss_10'], label='Entropy')
    ax3.plot(loss_data.index, loss_data['value_loss_10'], label='Value')
    ax3.plot(loss_data.index, loss_data['action_loss_10'], label='Action')
    ax3.tick_params(axis='y', labelcolor='blue')

    ax3.set_xlabel('Update', color='black')
    #ax3.set_ylim(-0.04, 0.1)

    if legend is True:
        ax3.legend()


    plt.title('Running average over previous 10 updates')
    #plt.show()
    if save == True:
        #plt.savefig(figure_file_loss)
        plt.close()
    return ax3

#%%



#%%


def plot_reward(
        lawn_num,
        run_id,
        reward_function,
        go_explore = 'False',
        save = False,
        ax = None
        ):




    csv_path = f"../PPO_logs/lawn{lawn_num}/go_explore_{go_explore}/reward_function{reward_function}/{run_id}.csv"

    reward_data = pd.read_csv(csv_path)


    #%%

    reward_data = pd.read_csv(csv_path, index_col = 'Run')
    reward_data['rewards_100'] = reward_data['Score'].rolling(100).mean()
    reward_data['rewards_100_max'] = reward_data['Score'].rolling(100).max()
    reward_data['perc_done_100'] = reward_data['Perc_done'].rolling(100).mean()
    reward_data['perc_done_100_max'] = reward_data['Perc_done'].rolling(100).max()
    #%%

    reward_data.dropna(inplace = True)
    #%%


    suptitle = f"Lawn: {lawn_num}  Run: {run_id}  Reward Function: {reward_function}  Go Explore: {go_explore}"

    if ax ==  None:
        fig, ax1 = plt.subplots()
    else:
        ax1 = ax

    ax1.set_ylabel('score', color='red')
    ax1.plot(reward_data.index, reward_data['rewards_100'], color='red', label = 'Score')
    ax1.plot(reward_data.index, reward_data['rewards_100_max'], color='orange', label = 'Score')
    ax1.tick_params(axis='y', labelcolor='red')

    ax1.set_xlabel('Run', color = 'black')


    #ax1.legend()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'blue'
    ax2.set_ylabel('% Done', color=color)  # we already handled the x-label with ax1
    ax2.plot(reward_data.index, reward_data['perc_done_100'], color=color)
    ax2.plot(reward_data.index, reward_data['perc_done_100_max'], color='indigo')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0,100)


    #plt.title('Running average over previous 100 runs')
    #plt.suptitle(suptitle)
    #plt.show()
    if save == True:
        #plt.savefig(figure_file_loss)
        plt.close()

    return ax1


#%%


def plot_frames(
        lawn_num,
        run_id,
        reward_function,
        go_explore = 'False',
        save = False,
        ax = None
        ):


    csv_path = f"../PPO_logs/lawn{lawn_num}/go_explore_{go_explore}/reward_function{reward_function}/{run_id}.csv"

    reward_data = pd.read_csv(csv_path)


    reward_data.dropna(inplace = True)
    I = reward_data['Perc_done'] == 100

    plt_data = reward_data[I]

    plt_data['Running_Min'] = plt_data['Frames'].cummin()

    x_max = max(plt_data.index)

    suptitle = f"Lawn: {lawn_num}  Run: {run_id}  Reward Function: {reward_function}  Go Explore: {go_explore}"

    if ax ==  None:
        fig, ax1 = plt.subplots()
    else:
        ax1 = ax

    ax1.set_ylabel('Log Frames', color='red')
    ax1.scatter(plt_data.index, plt_data['Frames'], color='red', label = 'Frames', s = 1)
    ax1.plot(plt_data.index, plt_data['Running_Min'] , color='orange', label = 'Running Minimum')
    ax1.tick_params(axis='y', labelcolor='red')

    ax1.set_xlabel('Run', color = 'black')
    ax1.set_xlim(0, x_max)

    ax1.set_ylim(400,800)


    #plt.title('Running average over previous 100 runs')
    #plt.suptitle(suptitle)
    #plt.show()
    if save == True:
        #plt.savefig(figure_file_loss)
        plt.close()

    return ax1
