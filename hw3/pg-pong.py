import numpy as np
import cPickle as pickle
import gym

env = gym.make("Pong-v0")
observation = env.reset() # frames form Pong?
#print observation


for _ in range(4000): #from 0 to 999
     env.render()
     action = env.action_space.sample()
     observation, reward, done, info = env.step(action)
     #print len(observation)

     #if reward != 0:
       #print reward
     #if done:
       #env.render()
       #break
