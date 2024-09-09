import retro
import time
import gym
from gym import Env
from gym.spaces import Box, MultiBinary
import numpy as np
import cv2
from matplotlib import pyplot as plt
# Import optuna for HPO
import optuna
# Import PPO for algos
from stable_baselines3 import PPO
# Evaluate Policy
from stable_baselines3.common.evaluation import evaluate_policy
# Import wrappers
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import os
from stable_baselines3.common.callbacks import BaseCallback

# print(retro.data.list_games())
retro.data.list_games()
env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
env.observation_space.sample()
env.action_space.sample()
obs = env.reset()
done = False
# for game in range(5):
#     while not done: 
#         if done: 
#             obs = env.reset()
#         env.render()
#         obs, reward, done, info = env.step(env.action_space.sample())
#         time.sleep(0.005)
#         print(reward)    
env.close()

# env.observation_space.sample()
# env.action_space.sample()

class StreetFighter(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        self.game = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', use_restricted_actions=retro.Actions.FILTERED)
        #self.score = 0
    
    def step(self, action):
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs)
        
        # Preprocess frame from game
        frame_delta = obs 
#         - self.previous_frame
#         self.previous_frame = obs 
        
        # Shape reward
        reward = info['score'] - self.score 
        self.score = info['score']

        return frame_delta, reward, done, info 
    
    def render(self, *args, **kwargs): 
        self.game.render()
    
    def reset(self):
        self.previous_frame = np.zeros(self.game.observation_space.shape)
        
        # Frame delta
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        
        # Create initial variables
        self.score = 0

        return obs
    
    def preprocess(self, observation): 
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84,84), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (84,84,1))
        return state
    
    def close(self): 
        self.game.close()

env = StreetFighter()
env.observation_space.shape

# for game in range(5):
#     while not done: 
#         if done: 
#             obs = env.reset()
#         env.render()
#         obs, reward, done, info = env.step(env.action_space.sample())
#         time.sleep(0.005)
#         print(reward)    
env.close()

LOG_DIR = './logs/'
OPT_DIR = './opt_nodelta/'

def optimize_ppo(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 8192),
        'gamma': trial.suggest_loguniform('gamma', 0.8, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-4),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, .99)
    }

def optimize_agent(trial):
    try:
        model_params = optimize_ppo(trial)
        env = StreetFighter()
        env = Monitor(env, LOG_DIR)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, 4, channels_order='last')
        model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **model_params)
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
        print(type(env.observation_space))
        return -1000
    
study = optuna.create_study(direction='maximize')
study.optimize(optimize_agent, n_trials=100, n_jobs=1)

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
    
CHECKPOINT_DIR = './train_nodelta/'
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

env = StreetFighter()
env = Monitor(env, LOG_DIR)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

model_params = {'n_steps': 2570.949, 'gamma': 0.906, 'learning_rate': 2e-07, 'clip_range': 0.369, 'gae_lambda': 0.891}
#model_params = {'n_steps': 8960, 'gamma': 0.906, 'learning_rate': 2e-03, 'clip_range': 0.369, 'gae_lambda': 0.891}
# model_params = study.best_params
model_params['n_steps'] = 40*64

model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, **model_params)
#model.load('./train_nodelta_backup/best_model_5460000.zip')
model.learn(total_timesteps=10000000, callback=callback)
env.close()

model = PPO.load('./train/best_model_90000')
env = StreetFighter()
env = Monitor(env, LOG_DIR)
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
        time.sleep(0.01)
        total_reward += reward
    print('Total Reward for episode {} is {}'.format(total_reward, episode))
    time.sleep(2)