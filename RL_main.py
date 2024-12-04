import torch
import numpy as np
from SHARED.model import *
from RL.environment import *
from stable_baselines3.common.env_checker import check_env
from SHARED.display_trajectories import print_metrics
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import SAC
from stable_baselines3 import *
from RL.helperFunctions import *
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from typing import Callable
import matplotlib.pyplot as plt
import time
from numpy import savetxt
from datetime import datetime
import sys

env = greenhouseEnv(use_growth_dif=False)
check_env(env)

if len(sys.argv) > 1:
    eps = sys.argv[1]
    GAMMA = sys.argv[2]

LR = 5e-3
stochastic = False
random_starts = False
if stochastic:
    num_simulations = 30
else:
    num_simulations = 1

# Training

# --- Schedules ---

job_desc = ""
# job_desc = "_G1"
current_time_path = "export_" + datetime.now().strftime("%Y%m%d_%H%M%S") + job_desc

modelType = "SAC"
if stochastic:
    inter_path = "stochastic/scale_" + str(noise_scale)
else:
    inter_path = "deterministic"

save_path = "export/" + current_time_path + "/models/" + modelType + "/" + inter_path
save_path_model = save_path
save_path_env = save_path

os.makedirs(save_path, exist_ok=True)

print(save_path_model)
print(save_path_env)

# ---

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func
  
# --- Model ---

env = greenhouseEnv(use_growth_dif=False, stochastic=True, random_starts=False)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs = True, norm_reward = False, clip_obs = 10.,gamma=GAMMA)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.05) * np.ones(n_actions))


policy_kwargs_sac = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[128,128], qf=[128,128]))
model = SAC(
    "MlpPolicy",
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

# --- Training ---

mean_reward = evaluate(model, num_episodes=5)
mean_reward = evaluate(model, num_episodes=5)
model.learn(total_timesteps=eps*max_steps,
             log_interval=10,
               progress_bar=True,
               callback=eval_callback)
mean_reward = evaluate(model, num_episodes=5)
env.save(save_path_env + "vecNormEnv.pkl")
model.save(save_path_model + 'last_model.zip')

# ------------------
# --- Evaluation ---
# ------------------

save_data = True

modelType = "SAC"

if True: # use last model
    save_path_model = save_path_model + "last_model.zip"
    # save_path_model = "models/" + modelType + "/" + inter_path + "last_model.zip"
else:
    save_path_model = "models/" + modelType + "/" + inter_path + "best_model.zip"
save_path_env = save_path_env + "vecNormEnv.pkl"
# save_path_env = "models/" + modelType + "/" + inter_path + "vecNormEnv.pkl"

# save_path_model = "models/SAC/stochastic/scale_0.2/best_model.zip"
# save_path_env   = "models/SAC/stochastic/scale_0.2/vecNormEnv.pkl"
print(save_path_model)
print(save_path_env)

# Actual used model in training
env.training = False
env_norm = greenhouseEnv(use_growth_dif=False, stochastic=stochastic)
env_norm = DummyVecEnv([lambda: env_norm])
env_norm = VecNormalize(env_norm, norm_obs = True, norm_reward = False, clip_obs = 10.,gamma=GAMMA,training = False)
# env_norm = env_norm.load(save_path_env, env_norm)
env_norm = VecNormalize.load(save_path_env, env_norm)
# env_norm.training = False

real_env = greenhouseEnv(use_growth_dif=False, stochastic= stochastic)


print('SAC')
model_loaded  = SAC.load(save_path_model,env=env_norm)
    
model_2_use =  model_loaded
print(model_2_use)

for sim_num in range(num_simulations):
    
    #Save directories
    if stochastic:
        file_path = "stochastic/scale_" + str(noise_scale)
    else:
        file_path = "deterministic"
    directory = 'results/RL/' + file_path + 'Sim_salim_' + str(sim_num) + '/'
    os.makedirs(directory, exist_ok=True)
    
    #Recorded outputs and states
    obs_log = []
    output_log = []

    #Initial conditions
    dry_mass_now=x0[0]
    dry_mass_next=0

    #Recorded value and reward recieved from env
    values = []
    total_reward=0
    rewards_log = []

    #Recorded rewards/penalites from evaluation
    cum_reward_log = [0]
    cum_penalties = [0]
    cost_log = []
    total_reward_eval=0
    total_penalty_eval=0

    #Recorded Computational time for control action
    comp_time_log = []

    #Reward evaluation function
    evaluate_rewards = partial(
        reward_evaluation,
        constraint_mins=np.array([C02_MIN_CONSTRAIN_MPC, TEMP_MIN_CONSTRAIN_MPC, HUM_MIN_CONSTRAIN]),
        constraint_maxs=np.array([C02_MAX_CONSTRAIN_MPC, TEMP_MAX_CONSTRAIN_MPC, HUM_MAX_CONSTRAIN])
    )

    #Simulating on real environment but using trained environment
    obs,_ = real_env.reset()
    obs_norm = env_norm.normalize_obs(obs)
    obs_log.append(obs)
    done = False
    
    while not done:
        
        #Interaction with environment
        timer = time.perf_counter()
        action, _states = model_2_use.predict(obs_norm, deterministic=True)
        timer = time.perf_counter() - timer
        obs, reward, done, _,info = real_env.step(action)
        
        dry_mass_next = info["output"][0]
        obs_norm = env_norm.normalize_obs(obs)

        #Reshaping for neural network value function 
        obs_tensor,_ = model_2_use.policy.obs_to_tensor(obs_norm)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)

        
        with torch.no_grad():
            vf1 = model_2_use.policy.critic.q_networks[1](torch.cat([obs_tensor,action_tensor], dim = 1))[0][0]
            # h = ent_coef * (find_H(obs_tensor,samples=100))
            vf2 = model_2_use.policy.critic.q_networks[0](torch.cat([obs_tensor,action_tensor], dim = 1))[0][0] #- h
            # print(h)
            vf = vf2
            values.append(vf)
        
        #Reward Evaluation
        u_opt = obs[4:7]
        reward_eval, penalties_eval = evaluate_rewards(delta_drymass=dry_mass_next - dry_mass_now,control_inputs=u_opt,
                            outputs2constrain=info['output'][1:]) 
        
        dry_mass_now = dry_mass_next   
        
        total_reward_eval += reward_eval
        total_penalty_eval += penalties_eval
        
        cum_reward_log.append(cum_reward_log[-1] + reward_eval - penalties_eval)
        cum_penalties.append(cum_penalties[-1] + penalties_eval)
        
        total_reward += reward
        rewards_log.append(total_reward)
        cost_log.append(reward)
        
        output_log.append(info['output'])
        comp_time_log.append(timer)
        obs_log.append(obs)
 
    # break
    
    #Reshaping
    obs_log         = np.vstack(obs_log)
    output_log      = np.vstack(output_log[:-1])
    values          = np.vstack(values[:-1]).squeeze(-1)
    rewards_log     = np.reshape(np.array([rewards_log[:-1]]),values.shape)
    comp_time_log   = np.array(comp_time_log)
    cum_reward_log  = np.array(cum_reward_log)
    
    U_log = obs_log[1:-1,4:7]
    Y_log = output_log
    D_log = obs_log[1:,8:]

    if save_data:
        savetxt(os.path.join(directory, 'Y_log.csv'), Y_log, delimiter=',')
        savetxt(os.path.join(directory, 'U_log.csv'), U_log, delimiter=',')
        savetxt(os.path.join(directory, 'D_log.csv'), D_log,delimiter=',')
        savetxt(os.path.join(directory, 'vf_log.csv'), values, delimiter=',')
        savetxt(os.path.join(directory, 'comp_time_log.csv'), comp_time_log, delimiter=',')
        savetxt(os.path.join(directory, 'rewards_log.csv'), cum_reward_log[:-1], delimiter=',')
        savetxt(os.path.join(directory, 'cost_log.csv'), cost_log, delimiter=',')

# ---

# plt.plot (Y_log[:,0])

# ---

'''TRAINING EVALUATION'''
print(f"Total performance recieved from environment {total_reward}")
print(f"Total reward recieved from evaluation {total_reward_eval}")
print(f"Total penalties recieved from evaluation {total_penalty_eval}")
print(f"Total evaluted performance recieved from environment {total_reward_eval-total_penalty_eval}")

print_metrics(
    Y_log, 
    U_log, 
    D_log, 
    vf=values, 
    rewards=rewards_log, 
    day_range=(0, 40), 
    time_log=comp_time_log,
    curr_time_path = current_time_path
)

print(rewards_log[-1])

# ------------------------------------------------------
# --- Duplicate .py and .slurm file to export folder ---
# ------------------------------------------------------

source_file_code = "RL_main3.py"
source_file_slurm = "RL_main.slurm"

destination_file_code = save_path + "/" + source_file_code
destination_file_slurm = save_path + "/" + source_file_slurm

# Copy destination_file_code
try:
    with open(source_file_code, 'rb') as src, open(destination_file_code, 'wb') as dst:
        dst.write(src.read())
    print(f"File copied to {destination_file_code} (destination_file_code)")
except FileNotFoundError:
    print("Source file or destination folder not found. (destination_file_code)")
except PermissionError:
    print("Permission denied. (destination_file_code)")
except Exception as e:
    print(f"An error occurred: {e} (destination_file_code)")

# Copy destination_file_slurm
try:
    with open(source_file_slurm, 'rb') as src, open(destination_file_slurm, 'wb') as dst:
        dst.write(src.read())
    print(f"File copied to {destination_file_slurm} (destination_file_slurm)")
except FileNotFoundError:
    print("Source file or destination folder not found. (destination_file_slurm)")
except PermissionError:
    print("Permission denied. (destination_file_slurm)")
except Exception as e:
    print(f"An error occurred: {e} (destination_file_slurm)")

