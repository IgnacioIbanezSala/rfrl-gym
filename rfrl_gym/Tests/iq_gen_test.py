import numpy as np
import rfrl_gym.datagen.modems as modems

class IQ_Gen:
    def __init__(self, history, entity):
        # Placeholder for data generation.
        
        self.history = history

        self.entity = entity

        self.samples_per_step = 10000
        self.noise_std = 0.01

        self.modem = self.__init_entity_modems(entity)

        self.fc = np.linspace(-0.5, 0.5, 2)+1/1/2
        self.rng=np.random.default_rng()

    def gen_iq(self):
        self.samples = np.roll(self.samples, self.samples_per_step, axis=0)
        #self.samples[0:self.samples_per_step] =  self.rng.normal(0.0, self.noise_std/np.sqrt(2.0), self.samples_per_step) + 1.0j*(self.rng.normal(0.0, self.noise_std/np.sqrt(2.0), self.samples_per_step))
        
        print(self.fc)
        cent_freq = self.rng.uniform(self.entity.modem_params['center_frequency'][0],self.entity.modem_params['center_frequency'][1])
        start = self.entity.modem_params['start']
        duration = self.entity.modem_params['duration']
        
        data = self.modem.gen_samps(int(duration*self.samples_per_step))
        self.t = np.linspace(0, int(duration*self.samples_per_step), int(duration*self.samples_per_step))
        data = data * np.exp(1j*2*np.pi*(self.fc[0]+cent_freq/1)*self.t)
        self.samples[int(start*self.samples_per_step):int((start+duration)*self.samples_per_step)] += data
        print(int(start*self.samples_per_step))
        print(int((start+duration)*self.samples_per_step))
        print(self.samples[int(start*self.samples_per_step):int((start+duration)*self.samples_per_step)])
        print(len(self.samples))
       
        return self.samples

    def reset(self):
        self.samples = self.rng.normal(0.0, self.noise_std/np.sqrt(2.0), self.samples_per_step*self.history) + 1.0j*(self.rng.normal(0.0, self.noise_std/np.sqrt(2.0), self.samples_per_step*self.history))

    # This function will create a dict that maps an entity within the scenario to its corresponding modem
    def __init_entity_modems(self, entity):
        entity_modems = dict()

        bandwidth = entity.modem_params['bandwidth']
        sps = int(1 / bandwidth)
        # Construct this entity's modem
        if entity.modem_params['type'] == 'psk' or entity.modem_params['type'] == 'qam' or entity.modem_params['type'] == 'ask':
            beta = 0.35
            span = 10
            trim = 0
            modem = modems.LDAPM(sps=sps, mod_type=entity.modem_params['type'], mod_order=entity.modem_params['order'], filt_type=entity.modem_params['filter'], beta=beta, span=span, trim=trim)
        else:
            modem = modems.Tone(sps=sps, mod_type=entity.modem_params['type'])

        return modem
