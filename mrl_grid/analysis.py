import matplotlib.pyplot as plt

def plot_avg_rewards(avg_episode_data):
    plt.plot(avg_episode_data['ep'], avg_episode_data['reward'], label="avg")
    plt.plot(avg_episode_data['ep'], avg_episode_data['reward_min'], label="min")
    plt.plot(avg_episode_data['ep'], avg_episode_data['reward_max'], label="max")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend(loc=4)
    plt.show()

def plot_rewards(episodes, rewards):
    plt.plot(range(episodes), rewards, color='orange')
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()

def plot_steps(episodes, steps):
    plt.plot(range(episodes), steps, color='purple')
    plt.xlabel("Episodes")
    plt.ylabel("No. of steps")
    plt.show()