import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from SHARED.model import *
from RL.environment import *
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3 import *
from RL.helperFunctions import *
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from typing import Callable
import matplotlib.pyplot as plt
import os

# Initialize environment
env = greenhouseEnv(use_growth_dif=False)
check_env(env)

GAMMA = 0.95
LR = 5e-3 
stochastic = False
random_starts = False
num_simulations = 30 if stochastic else 1

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        """
        return progress_remaining * initial_value

    return func
  
def exp_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        """
        return (progress_remaining**2) * initial_value

    return func
  
def customCallback(env, savePath):
    env.envs[0].training = False
    EvalCallback(env, best_model_save_path=savePath,
                log_path="logs/", eval_freq=max_steps*2,
                n_eval_episodes=1, deterministic=True,
                render=False,warn=False,verbose=0)
    env.envs[0].training = True

# Setup model paths
modelType = "SAC"
inter_path = "stochastic/scale_" + str(noise_scale) +'/' if stochastic else "deterministic/"
save_path_model = "models/" + modelType + "/" + inter_path
save_path_env = "models/" + modelType + "/" + inter_path + "vecNormEnv.pkl"
os.makedirs(save_path_model, exist_ok=True)

# Initialize environment and model
env = greenhouseEnv(use_growth_dif=False, stochastic=True, random_starts=False)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.,gamma=GAMMA)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.05) * np.ones(n_actions))

if modelType.lower() == "ppo":
    policy_kwargs_ppo = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[128,128], vf=[128,128]))
    model = PPO("MlpPolicy",env,verbose=0, policy_kwargs=policy_kwargs_ppo,gamma=GAMMA,
               gae_lambda=0.95,max_grad_norm=0.5,tensorboard_log="logs/",
               learning_rate=linear_schedule(LR))
elif modelType.lower() == "sac":
    policy_kwargs_sac = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[128,128], qf=[128,128]))
    model = SAC("MlpPolicy",
        env,
        verbose=0,
        gamma=GAMMA,
        policy_kwargs=policy_kwargs_sac, 
        learning_starts=warm_up_eps*max_steps,
        batch_size=1024,
        tensorboard_log="logs/",
        learning_rate=linear_schedule(LR),
        buffer_size=100000
    )

eval_callback = EvalCallback(env, best_model_save_path=save_path_model,
                           log_path="logs/", eval_freq=max_steps*2,
                           n_eval_episodes=5, deterministic=True,
                           render=False,warn=False,verbose=0)

# Training
mean_reward = evaluate(model, num_episodes=5)
model.learn(total_timesteps=eps*max_steps, log_interval=10, progress_bar=True,callback=eval_callback)
mean_reward = evaluate(model, num_episodes=5)
env.save(save_path_env)
model.save(save_path_model + 'last_model.zip')