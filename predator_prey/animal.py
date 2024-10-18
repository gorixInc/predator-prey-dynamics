import numpy as np
from math import sqrt, pi, cos, sin, atan2
import mesa
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
import matplotlib.pyplot as plt
from mesa.time import BaseScheduler
from numpy import inf

class Animal(Agent):
    def __init__(
        self,
        model,
        max_speed = 20,
        acceleration_factor = 5,
        heading = 0,
        size = 1,
        vision_range = np.inf,
        conspecific_vision_cap = 4,
        heterospecific_vision_cap = 4,
        n_messages = 0,
        hidden_sizes = [],
        initial_weights_disp = 0.1,
        self_obs_size = 2, 
        other_obs_size = 4,
        species = 'default'
    ):
        super().__init__(self, model)


        self.speed = 0
        self.speed_x = 0
        self.speed_y = 0 
        self.max_speed = max_speed
        self.init_heading = heading
        self.heading = heading
        self.size = size

        self.acceleration = 0
        self.acceleration_factor = acceleration_factor

        self.messages = np.zeros(n_messages)
        self.vision_range = vision_range
        self.conspecific_vision_cap = conspecific_vision_cap
        self.heterospecific_vision_cap = heterospecific_vision_cap
        self.species = species
        self.fitness = 0
        self.generation = 0
        self.run = 0
        self.alive = True

        self.self_obs_size = self_obs_size
        self.consp_obs_size = other_obs_size + len(self.messages)
        self.hetero_obs_size = other_obs_size
        
        self.full_obs_size =  self.self_obs_size + \
                              self.consp_obs_size * self.conspecific_vision_cap + \
                              self.hetero_obs_size * self.heterospecific_vision_cap
        self.full_act_size = 2 + len(self.messages)


        self.hidden_sizes = hidden_sizes
        self.layer_sizes = [self.full_obs_size] + self.hidden_sizes + [self.full_act_size]
        self.Ws, self.bs = [], []
        for i in range(len(self.layer_sizes) - 1):
            W = np.random.normal(0, initial_weights_disp, (self.layer_sizes[i], self.layer_sizes[i+1]))
            b = np.zeros(self.layer_sizes[i+1])
            self.Ws.append(W)
            self.bs.append(b)

    def reset(self):
        self.heading = self.init_heading
        self.speed = 0
        self.alive = True

    def step(self):
        if not self.alive:
            return
        neighbors = self.model.space.get_neighbors(self.pos, self.vision_range, include_center=False)
        observations = self.observe(neighbors)
        self.act(neighbors, observations)
    
    def get_actions(self, observations):
        X = observations
        for i in range(len(self.Ws)): 
            X = X @ self.Ws[i] + self.bs[i]
        Y = np.clip(X, -1, 1)
        return Y

    def act(self, neighbours, observations): 
        Y = self.get_actions(observations)
        accel_x, accel_y = Y[0], Y[1] 
        message = Y[2:].astype(int)

        self.message = message
        self.speed_x += accel_x * self.acceleration_factor * self.model.dt
        self.speed_y += accel_y * self.acceleration_factor * self.model.dt
        self.speed = np.sqrt(self.speed_x**2 + self.speed_y**2)

        if self.speed > self.max_speed:
            speed_diff = self.speed / self.max_speed
            self.speed_x = self.speed_x/speed_diff
            self.speed_y = self.speed_y/speed_diff
            self.speed = self.speed / speed_diff

        dx = self.speed_x * self.model.dt
        dy = self.speed_y * self.model.dt
        new_pos = self.pos + np.array([dx, dy])
        self.model.space.move_agent(self, new_pos)
    
    def observe(self, neighbours):

        self_obs = np.zeros(self.self_obs_size)
        consp_obs = np.zeros(self.consp_obs_size * self.conspecific_vision_cap)
        hetero_obs = np.zeros(self.hetero_obs_size * self.heterospecific_vision_cap)

        self_obs = np.array([self.speed_x/self.max_speed,  self.speed_y/self.max_speed])

        neighbours = np.array(neighbours)
        neighb_species = np.array([n.species for n in neighbours])
        consp_inds = np.where(neighb_species == self.species)[0]
        hetero_inds = np.where(neighb_species != self.species)[0]

        for i, ind in enumerate(consp_inds):
            if i == self.conspecific_vision_cap:
                break
            consp = neighbours[ind]

            delta_x, delta_y = (self.pos[0] - consp.pos[0])/self.vision_range, (self.pos[1] - consp.pos[1])/self.vision_range,

            speed_obs = [consp.speed_x/consp.max_speed, consp.speed_y]
            this_obs = np.array([delta_x, delta_y] + speed_obs + list(consp.messages)) # size 2 + 2 + n_messages
            consp_obs[i*self.consp_obs_size:(i+1)*self.consp_obs_size] = this_obs

            
        for i, ind in enumerate(hetero_inds):
            if i == self.heterospecific_vision_cap:
                break
            hetero = neighbours[ind]
            delta_x, delta_y = (self.pos[0] - hetero.pos[0])/self.vision_range, (self.pos[1] - hetero.pos[1])/self.vision_range

            speed_obs = [hetero.speed_x, hetero.speed_y]
            this_obs = np.array([delta_x, delta_y] + speed_obs) # size 2 + 2
            hetero_obs[i*self.hetero_obs_size:(i+1)*self.hetero_obs_size] = this_obs

        all_observations = np.hstack((self_obs, consp_obs, hetero_obs))
        return all_observations 