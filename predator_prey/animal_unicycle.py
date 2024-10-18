from predator_prey.animal import Animal
import numpy as np
from math import pi, sqrt

class AnimalUnicycle(Animal):
    def __init__(self, *args, **kwargs):
        super().__init__(self_obs_size=2, other_obs_size=3, *args, **kwargs)

    def act(self, neighbours, observations): 
        Y = self.get_actions(observations)
        
        acceleration, self.heading = Y[0], Y[1] 
        message = Y[2:].astype(int)

        self.message = message
        self.speed += acceleration * self.acceleration_factor * self.model.dt
        self.speed = np.clip(self.speed, -self.max_speed, self.max_speed) 

        dx = np.cos(self.heading * pi) * self.speed * self.model.dt
        dy = np.sin(self.heading * pi) * self.speed * self.model.dt
        new_pos = self.pos + np.array([dx, dy])
        self.model.space.move_agent(self, new_pos)

    def observe(self, neighbours):

        self_obs = np.zeros(self.self_obs_size)
        consp_obs = np.zeros(self.consp_obs_size * self.conspecific_vision_cap)
        hetero_obs = np.zeros(self.hetero_obs_size * self.heterospecific_vision_cap)

        self_obs = np.array([self.heading,  self.speed/self.max_speed])

        neighbours = np.array(neighbours)
        neighb_species = np.array([n.species for n in neighbours])
        consp_inds = np.where(neighb_species == self.species)[0]
        hetero_inds = np.where(neighb_species != self.species)[0]

        for i, ind in enumerate(consp_inds):
            if i == self.conspecific_vision_cap:
                break
            consp = neighbours[ind]
            relative_heading = consp.heading - self.heading

            delta_x, delta_y = (self.pos[0] - consp.pos[0])/self.vision_range, (self.pos[1] - consp.pos[1])/self.vision_range,
            distance = sqrt(delta_x**2 + delta_y**2)

            #speed_obs = [consp.speed_x, consp.speed_y]
            this_obs = np.array([relative_heading, distance, consp.speed/consp.max_speed] + list(consp.messages)) # size 2 + 1 + n_messages
            consp_obs[i*self.consp_obs_size:(i+1)*self.consp_obs_size] = this_obs

            
        for i, ind in enumerate(hetero_inds):
            if i == self.heterospecific_vision_cap:
                break
            hetero = neighbours[ind]
            relative_heading = hetero.heading - self.heading
            
            delta_x, delta_y = (self.pos[0] - hetero.pos[0])/self.vision_range, (self.pos[1] - hetero.pos[1])/self.vision_range
            distance = sqrt(delta_x**2 + delta_y**2)

            this_obs = np.array([relative_heading, distance, hetero.speed/hetero.max_speed]) # size 3
            hetero_obs[i*self.hetero_obs_size:(i+1)*self.hetero_obs_size] = this_obs

        all_observations = np.hstack((self_obs, consp_obs, hetero_obs))
        return all_observations 
        