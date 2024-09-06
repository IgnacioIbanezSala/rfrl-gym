import gymnasium as gym
import json
import argparse
import matplotlib.pyplot as plt
import rfrl_gym
import math


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scenario', default='sb3_test_scenario.json', type=str, help='The scenario file to preview in the RFRL gym environment.')
parser.add_argument('-m', '--gym_mode', default='abstract', type=str, help='Which type of RFRL gym environment to run.')
parser.add_argument('-e', '--epochs', default=1000, type=int, help='Number of training epochs.')
args = parser.parse_args()

if args.gym_mode == 'abstract':
    env = gym.make('rfrl-gym-abstract-v0', scenario_filename=args.scenario)
elif args.gym_mode == 'iq':
    env = gym.make('rfrl-gym-iq-v0', scenario_filename=args.scenario)
env.reset()

f_idx = open('scenarios/' + args.scenario)
scenario_metadata = json.load(f_idx)



N = 100
Num_Channels = scenario_metadata['environment']['num_channels']
channels_sellected = []
numbers_of_selections = [0] * Num_Channels
sums_of_rewards = [0] * Num_Channels
total_reward = 0


obs, info = env.reset()
terminated = truncated= False
running_reward = 0
rewards = []


while not terminated and not truncated:
    for n in range(0, N):
        action = 0
        max_upper_bound = 0
        for i in range(0, Num_Channels):
            if(numbers_of_selections[i] > 0):
                average_reward = sums_of_rewards[i] / numbers_of_selections[i]
                #delta_i = math.sqrt(math.log(n) / numbers_of_selections[i])
                delta_i = math.sqrt(math.log(n/numbers_of_selections[i]))
                upper_bound = average_reward + delta_i
            else:
                upper_bound = 1e400
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                action = i
        channels_sellected.append(action)
        numbers_of_selections[action] = numbers_of_selections[action] + 1
        obs, reward, terminated, truncated, info = env.step(action)
        sums_of_rewards[action] = sums_of_rewards[action] + reward
        total_reward = total_reward + reward
        
      
    
env.render()

env.close()