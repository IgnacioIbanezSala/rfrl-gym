import gymnasium as gym
import json
import argparse
import matplotlib.pyplot as plt
import rfrl_gym
import math

import rfrl_gym.entities


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

Num_Channels = scenario_metadata['environment']['num_channels']
Samples = []
#Entity = rfrl_gym.entities(entity_label = scenario_metadata['entities'], num_channels = Num_Channels, )

entity_idx = 0
entity_list = []
for entity in scenario_metadata['entities']:  
    entity_idx += 1
    obj_str = 'rfrl_gym.entities.' + scenario_metadata['entities'][entity]['type'] + '(entity_label=\'' + str(entity) + '\', num_channels=' + str(Num_Channels) + ', '
    for param in scenario_metadata['entities'][entity]:
        if not param == 'type':
            obj_str += (param + '=' + str(scenario_metadata['entities'][entity][param]) + ', ')
    obj_str += ')'
    entity_list.append(eval(obj_str))      
    entity_list[-1].set_entity_index(entity_idx)
num_entities = len(entity_list)

action_list = [num_entities]

iq_gen = rfrl_gym.datagen.iq_gen.IQ_Gen(Num_Channels, num_entities, [], entity_list)

entity_idx = 0
for entity in entity_list:
            entity_idx += 1
            entity_action = entity.get_action()
            action_list[num_entities] = entity_action
            
Samples = iq_gen.gen_iq(action_list)

plt.plot(Samples) 
plt.show()