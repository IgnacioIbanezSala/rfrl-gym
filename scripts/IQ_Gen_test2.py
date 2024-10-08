import gymnasium as gym
import json
import argparse
import matplotlib.pyplot as plt
import rfrl_gym
import math
import numpy as np

import rfrl_gym.Tests
import rfrl_gym.entities


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scenario', default='entity_test.json', type=str, help='The scenario file to preview in the RFRL gym environment.')
parser.add_argument('-m', '--gym_mode', default='abstract', type=str, help='Which type of RFRL gym environment to run.')
parser.add_argument('-e', '--epochs', default=1000, type=int, help='Number of training epochs.')
parser.add_argument('-t', '--title', default="IQ_gen_test", type=str, help='Title of image to save.')
args = parser.parse_args()

if args.gym_mode == 'abstract':
    env = gym.make('rfrl-gym-abstract-v0', scenario_filename=args.scenario)
elif args.gym_mode == 'iq':
    env = gym.make('rfrl-gym-iq-v0', scenario_filename=args.scenario)
env.reset()

f_idx = open('scenarios/' + args.scenario)
scenario_metadata = json.load(f_idx)

img_title = args.title + ".png"


Num_Channels = scenario_metadata['environment']['num_channels']

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

name = obj_str

num_entities = len(entity_list)
print(num_entities)


action_list = [None] * num_entities

iq_gen = rfrl_gym.Tests.iq_gen_test.IQ_Gen(scenario_metadata['render']['render_history'], entity_list[0])
iq_gen.reset()

info = { }
info["step_number"] = 0
info["Samples"] = np.zeros(10000)

for entity in entity_list:
            entity.reset(info)     


info["Samples"] = iq_gen.gen_iq()
print(info["Samples"][0:10000])
figure, axis = plt.subplots(1, 3)

axis[0].plot(np.real(info["Samples"][2500:5000]))
axis[0].set_title(name, loc='center', wrap=True)
axis[1].plot(np.imag(info["Samples"][2500:5000]))
axis[1].set_title("imag", loc='center', wrap=True)
axis[2].plot(np.real(info["Samples"][2500:5000]), np.imag(info["Samples"][2500:5000]), '.')
axis[2].set_title("const", loc='center', wrap=True)
plt.savefig(img_title)
plt.show()