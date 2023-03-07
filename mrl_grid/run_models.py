import os
import shutil
from stable_baselines3 import DQN, PPO, A2C

TIMESTEPS = 10_000

def run_ppo(env, models_dir, logdir, episodes = 100_000):
    set_folders(models_dir, logdir)
    delete_files(models_dir)
    delete_folder(f"{logdir}/PPO_0")

    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
    split_num = round(episodes / TIMESTEPS)
    for i in range(1, split_num+1):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{models_dir}/{TIMESTEPS*i}") 

def run_dqn(env, models_dir, logdir, episodes = 100_000):
    set_folders(models_dir, logdir)
    delete_files(models_dir)
    delete_folder(f"{logdir}/DQN_0")

    model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
    split_num = round(episodes / TIMESTEPS)
    for i in range(1, split_num+1):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN")
        model.save(f"{models_dir}/{TIMESTEPS*i}")
   
def run_a2c(env, models_dir, logdir, episodes = 100_000):
    set_folders(models_dir, logdir)
    delete_files(models_dir)
    delete_folder(f"{logdir}/A2C_0")

    model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
    split_num = round(episodes / TIMESTEPS)
    for i in range(1, split_num+1):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
        model.save(f"{models_dir}/{TIMESTEPS*i}")

def load_model(model_path, env, model_type):
    if model_type == "PPO":
        model = PPO.load(model_path, env=env)
    if model_type == "A2C":
        model = A2C.load(model_path, env=env)
    if model_type == "DQN":
        model = DQN.load(model_path, env=env)
    
    episodes = 5
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            env.render()
            action, states = model.predict(obs)
            obs, reward, done, info = env.step(action)

def set_folders(models_dir: str, logdir: str):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

def delete_files(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path) and filename != "monitor.csv":
            os.remove(file_path)


def delete_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)