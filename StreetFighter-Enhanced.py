import retro
import time
from gym import Env
from gym.spaces import Box, MultiBinary
import numpy as np
import cv2
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import os
import optuna
from stable_baselines3.common.callbacks import BaseCallback

# Custom environment class for Street Fighter
class StreetFighter(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        self.game = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', use_restricted_actions=retro.Actions.FILTERED)
        self.health = 176
        self.enemy_health = 176
        self.time = 99
    
    def step(self, action):
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs)
        
        # Health delta-based reward shaping
        health_delta = (self.enemy_health - info['enemy_health']) * 2 + (info['health'] - self.health)
        time_penalty = (self.time - info['time']) * 0.1  # Add small penalty for longer episodes
        
        reward = health_delta + time_penalty
        self.health = info['health']
        self.enemy_health = info['enemy_health']
        self.time = info['time']
        
        return obs, reward, done, info
    
    def render(self, *args, **kwargs): 
        self.game.render(*args, **kwargs)
    
    def reset(self):
        self.previous_frame = np.zeros(self.game.observation_space.shape)
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.health = 176
        self.enemy_health = 176
        self.time = 99
        return obs
    
    def preprocess(self, observation): 
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84,84), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (84,84,1))
        return state
    
    def close(self): 
        self.game.close()

# Logging directory
LOG_DIR = './logs/'
OPT_DIR = './opt_hyperparameters/'

# Hyperparameter optimization using Optuna
def optimize_ppo(trial):
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 8192),
        'gamma': trial.suggest_loguniform('gamma', 0.8, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-4),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, .99),
        'ent_coef': trial.suggest_loguniform('ent_coef', 0.0001, 0.01),  # Added entropy coefficient
        'vf_coef': trial.suggest_loguniform('vf_coef', 0.1, 1.0)  # Added value function coefficient
    }

# Adding different RL architectures
def select_algorithm(trial, env):
    alg_name = trial.suggest_categorical('algorithm', ['PPO', 'A2C', 'DQN'])
    if alg_name == 'PPO':
        return PPO('CnnPolicy', env, verbose=0, **optimize_ppo(trial))
    elif alg_name == 'A2C':
        return A2C('CnnPolicy', env, verbose=0, **optimize_ppo(trial))
    elif alg_name == 'DQN':
        return DQN('CnnPolicy', env, verbose=0, **optimize_ppo(trial))

def optimize_agent(trial):
    try:
        env = StreetFighter()
        env = Monitor(env, LOG_DIR)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, 4, channels_order='last')

        model = select_algorithm(trial, env)
        model.learn(total_timesteps=100000)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=20)
        env.close()

        if not os.path.exists(OPT_DIR):
            os.makedirs(OPT_DIR)

        SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(trial.number))
        model.save(SAVE_PATH)
        return mean_reward
    except Exception as e: 
        print(f"Error in trial {trial.number}: {e}")
        return -1000

# Start Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(optimize_agent, n_trials=100, n_jobs=1)

# Callback for saving model checkpoints
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, trial_number, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.trial_number = trial_number

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_trial_{}_{}'.format(self.trial_number,self.n_calls))
            self.model.save(model_path)
        return True

# Define training and logging callback
CHECKPOINT_DIR = './train_checkpoints/'
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

# Final training using the best parameters
env = StreetFighter()
env = Monitor(env, LOG_DIR)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

# Get the best hyperparameters from Optuna
model_params = study.best_params
model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, **model_params)
model.learn(total_timesteps=10000000, callback=callback)
env.close()

# Evaluation of the model
model = PPO.load('./train_checkpoints/best_model_90000')
env = StreetFighter()
env = Monitor(env)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()

for episode in range(1): 
    obs = env.reset()
    done = False
    total_reward = 0
    while not done: 
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        total_reward += reward
    print('Total Reward for episode {} is {}'.format(episode, total_reward))
    time.sleep(2)
