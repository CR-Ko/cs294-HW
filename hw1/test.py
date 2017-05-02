#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import matplotlib.pyplot as plt
from gym import wrappers

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')
    #print policy_fn
    print('Start to test pkl')
    f = open('/home/koo/hw1/experts/Hopper-v1.pkl','rb')
    info = pickle.load(f)         
    #print info

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        #env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1')
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            print(env.observation_space)
            print(env.action_space)
            #print(env.observation_space.high)
            #print(env.observation_space.low)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                #What I want 
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                
                obs, r, done, _ = env.step(action)

                #np.savez('testdata.npz', observations=np.array(observations), actions=np.array(actions))
                #data = np.load('traindata.npz') 
                #print data.files 
                #x = data['observations'] 
                #y = data['actions']
                #print('observations',x)
                #print('actions',y)   
                #print('xshape',x.shape)
                #print('yshape',y.shape) 
                ###############################################
                totalr += r
                steps += 1
                #Range of the action values  
                #mins = np.min(actions, 0)
                #maxs = np.max(actions, 0)
                #print(mins)
                #print(maxs)

                #plt.figure(1)
                #plt.plot(mins)
                #plt.plot(maxs) 
                if args.render:
                    env.render()
                if steps % 100 == 0: 
                      print("%i/%i"%(steps, max_steps))
                     # print("obs",obs)
                     # print("action",action)
                     # print('observations',observations)
                     # print("actions",actions)
                if steps >= max_steps:
                    break
            returns.append(totalr)
           
        print('returns', returns) 
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

if __name__ == '__main__':
    main()
