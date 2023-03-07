from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    'DQN': 'purple',
    'PPO': 'orange',
    'A2C': 'blue',
    'NNA': 'grey',
}

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(model_types, grid_names, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    fig = plt.figure(title)
    fig.set_size_inches(12, 5)

    for model_type, grid_name in zip(model_types, grid_names):
        log_folder = f"models/{grid_name}/{model_type}/"
        x, y = ts2xy(load_results(log_folder), 'timesteps')
        y = moving_average(y, window=50)
        x = x[len(x) - len(y):] # Truncate x
        color = COLORS.get(model_type, 'black')
        plt.plot(x, y, label=model_type, color=color)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.legend()
    plt.savefig(f'images/{title}.png')
    plt.show()

if __name__ == '__main__':
    grid_name = "10x10"
    title = f"{grid_name} Grid Learning Curve"
    grid_names = [grid_name]
    model_types = ["PPO"]

    plot_results(model_types, grid_names, title)