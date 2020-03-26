# -*- coding: utf-8 -*-

import numpy as np
import pylab as plt
import random

class sets(object):
    """
    The construction process to build a single trial while reporting the
    variables present for posterior analysis

    """

    def __init__(self):

        self.timesteps = 600
        self.n_tasks = 6
        self.fixation = 1
        self.units = 32
        self.modalities = 2




    def create_set(self, n_task, stim1_dir, stim1_mod, Tstim1, Tgo, stim2_dir = 0,
        stim2_mod = 0, Tdelay = 0, Tstim2 = 0):

        """
        Generates the input and output for any given trial dependent on the
        task that is being considered

        """

        X = np.zeros((self.fixation +
            self.units*self.modalities +
            self.n_tasks, self.timesteps))

        Y = np.zeros((self.fixation +
            self.units, self.timesteps))

        pos = self.n_tasks # modality 1 units start after rule units

        if n_task in [1,4]: # for task RT
           X[0,:] = 1

        X[0,:Tgo] = 1 # fixation
        X[1 + n_task,:] = 1 # task signal
        X[pos + (self.units*stim1_mod) + stim1_dir, :Tstim1] = 1
        X[pos + (self.units*stim2_mod) + stim2_dir,
         Tstim1+Tdelay:Tstim1+Tdelay+Tstim2 ] = 1
        noise = np.random.normal(scale = 0.1, size = X.shape)
        X = X + noise

        Y[0,:Tgo] = 1 # fixated
        if n_task in [0,1,2]: # go tasks
            Y[stim1_dir,Tgo:] = 1
        elif n_task in [3,4,5]: # anti tasks
            response = stim1_dir -  int(self.units/2.0)
            if  response < 1:
                response = self.units + response
            Y[response,Tgo:] = 1

        return X, Y


    def go_task(self):
        """
        Generates a trial from a Go task where the output follows the direction
        of the stimulus input when it ceases to exist

        """
        task = 0
        stim_dir = np.stack(random.sample(range(1, self.units+1), 1))
        stim_mod = np.stack(random.sample(range(0,2), 1)) # 0 for mod1, 1 for mod2
        Tstim1 = int(np.random.uniform(100,500))

        X,Y = self.create_set(n_task = task, stim1_dir = stim_dir, stim1_mod = stim_mod, Tstim1 = Tstim1, Tgo = Tstim1)

        data = {
                'X': X,
                'Y': Y,
                'stim1_dir': stim_dir,
                'stim1_mod': stim_mod,
                'Tstim1': Tstim1,
                'Tdelay': None,
                'stim2_dir': None,
                'stim2_mod': None,
                'Tstim2': None,
                'Tgo': Tstim1,
                'task': 'go'
        }


        return data

    def RT_go_task(self):
        """
        Generates a trial from a Reaction Time Go task where the output follows
        the direction of the stimulus input from the moment the stimulus is
        presented
        """

        task = 1
        stim_dir = np.stack(random.sample(range(1, self.units+1), 1))
        stim_mod = np.stack(random.sample(range(0,2), 1)) # 0 for mod1, 1 for mod2
        Tstim1 = int(np.random.uniform(100,500))

        X,Y = self.create_set(n_task = task, stim1_dir = stim_dir, stim1_mod = stim_mod, Tstim1 = Tstim1, Tgo = 0)

        data = {
                'X': X,
                'Y': Y,
                'stim1_dir': stim_dir,
                'stim1_mod': stim_mod,
                'Tstim1': Tstim1,
                'Tdelay': None,
                'stim2_dir': None,
                'stim2_mod': None,
                'Tstim2': None,
                'Tgo': 0,
                'task': 'rt_go'
        }

        return data

    def Dly_go_task(self):
        """
        Generates a trial from a Delay Go task where the output follows the direction
        of the stimulus input when fixation drops to 0, moments after the stimulus
        disappears
        """


        task = 2
        dly_samples = [50,100,150,200,250]

        stim_dir = np.stack(random.sample(range(1, self.units+1), 1))
        stim_mod = np.stack(random.sample(range(0,2), 1)) # 0 for mod1, 1 for mod2
        Tstim1 = int(np.random.uniform(100,300))

        Tdelay = np.random.choice(dly_samples)

        X,Y = self.create_set(n_task = task, stim1_dir = stim_dir, stim1_mod = stim_mod, Tstim1 = Tstim1, Tdelay = Tdelay, Tgo = Tstim1 + Tdelay)


        data = {
                'X': X,
                'Y': Y,
                'stim1_dir': stim_dir,
                'stim1_mod': stim_mod,
                'Tstim1': Tstim1,
                'Tdelay': Tdelay,
                'stim2_dir': None,
                'stim2_mod': None,
                'Tstim2': None,
                'Tgo': Tstim1 + Tdelay,
                'task': 'dly_go'
        }

        return data

    def anti_task(self):
        """
        Generates a trial from an Anti task where the output goes against the
        direction of the stimulus input when it ceases to exist
        """
        task = 3
        stim_dir = np.stack(random.sample(range(1, self.units+1), 1))
        stim_mod = np.stack(random.sample(range(0,2), 1)) # 0 for mod1, 1 for mod2
        Tstim1 = int(np.random.uniform(100,500))

        X,Y = self.create_set(n_task = task, stim1_dir = stim_dir, stim1_mod = stim_mod, Tstim1 = Tstim1, Tgo = Tstim1)


        data = {
                'X': X,
                'Y': Y,
                'stim1_dir': stim_dir,
                'stim1_mod': stim_mod,
                'Tstim1': Tstim1,
                'Tdelay': None,
                'stim2_dir': None,
                'stim2_mod': None,
                'Tstim2': None,
                'Tgo': Tstim1,
                'task': 'anti'
        }

        return data

    def RT_anti_task(self):
        """
        Generates a trial from a Reaction Time Go task where the output goes
        against the direction of the stimulus input from the moment the stimulus
        is presented
        """

        task = 4
        stim_dir = np.stack(random.sample(range(1, self.units+1), 1))
        stim_mod = np.stack(random.sample(range(0,2), 1)) # 0 for mod1, 1 for mod2
        Tstim1 = int(np.random.uniform(100,500))

        X,Y = self.create_set(n_task = task, stim1_dir = stim_dir, stim1_mod = stim_mod, Tstim1 = Tstim1, Tgo = 0)

        data = {
                'X': X,
                'Y': Y,
                'stim1_dir': stim_dir,
                'stim1_mod': stim_mod,
                'Tstim1': Tstim1,
                'Tdelay': None,
                'stim2_dir': None,
                'stim2_mod': None,
                'Tstim2': None,
                'Tgo': 0,
                'task': 'rt_anti'
        }

        return data

    def Dly_anti_task(self):
        """
        Generates a trial from a Delay Go task where the output goes against the
        direction of the stimulus input when fixation drops to 0, moments after
        the stimulus disappears
        """


        task = 5
        dly_samples = [50,100,150,200,250]

        stim_dir = np.stack(random.sample(range(1, self.units+1), 1))
        stim_mod = np.stack(random.sample(range(0,2), 1)) # 0 for mod1, 1 for mod2
        Tstim1 = int(np.random.uniform(100,300))

        Tdelay = np.random.choice(dly_samples)

        X,Y = self.create_set(n_task = task, stim1_dir = stim_dir, stim1_mod = stim_mod, Tstim1 = Tstim1, Tdelay = Tdelay, Tgo = Tstim1 + Tdelay)


        data = {
                'X': X,
                'Y': Y,
                'stim1_dir': stim_dir,
                'stim1_mod': stim_mod,
                'Tstim1': Tstim1,
                'Tdelay': Tdelay,
                'stim2_dir': None,
                'stim2_mod': None,
                'Tstim2': None,
                'Tgo': Tstim1 + Tdelay,
                'task': 'dly_anti'
        }

        return data
