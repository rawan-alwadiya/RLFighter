import retro
from gym import Env
from gym.spaces import Discrete, Box, MultiBinary
import numpy as np
import cv2
import time
# Import PPO for algos
from stable_baselines3 import PPO
# Evaluate Policy
from stable_baselines3.common.evaluation import evaluate_policy
# Import Wrappers
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

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
        # reward = info['score'] - self.score 
        # self.score = info['score']
        reward = (self.enemy_health - info['enemy_health'])*2 + (info['health'] - self.health)

        return frame_delta, reward, done, info 
    
    def render(self, *args, **kwargs): 
        self.game.render(*args, **kwargs)
    
    def reset(self):
        self.previous_frame = np.zeros(self.game.observation_space.shape)
        
        # Frame delta
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        self.health = 176
        self.enemy_health = 176
        
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

#5.46m is the model that performed best 6.16m pretty good as well
model = PPO.load('./train_nodelta/best_model_770000.zip')
#model = PPO.load('./train_nodelta/best_model_170000.zip')

env = StreetFighter()
env = Monitor(env)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

for episode in range(1): 
    obs = env.reset()
    done = False
    total_reward = 0
    while not done: 
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        # time.sleep(0.01)
        total_reward += reward
    print('Total Reward for episode {} is {}'.format(total_reward, episode))
    time.sleep(2)

env.close()    