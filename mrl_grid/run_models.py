import os
from stable_baselines3 import DQN, PPO, A2C

TIMESTEPS = 10_000

def run_ppo(env, models_dir, logdir, episodes = 100_000):
    set_folders(models_dir, logdir)

    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
    split_num = round(episodes / TIMESTEPS)
    for i in range(1, split_num):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{models_dir}/{TIMESTEPS*i}") 
   
def run_a2c(env, models_dir, logdir, episodes = 100_000):
    set_folders(models_dir, logdir)

    model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
    split_num = round(episodes / TIMESTEPS)
    for i in range(1, split_num):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
        model.save(f"{models_dir}/{TIMESTEPS*i}")

def set_folders(models_dir: str, logdir: str):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)