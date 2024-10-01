"""A mesa implementation of our predator and prey agents
"""

import numpy as np
from math import sqrt, pi, cos, sin
import mesa



class Animal(mesa.Agent):
    def __init__(
        self,
        model,
        speed = 0,
        heading = 0,
        vision_range = np.inf,
        conspecific_vision_cap = 4,
        heterospecific_vision_cap = 4,
        n_messages = 0,
        species = 'default'

    ):
        super().__init__(self, model)

        self.speed = speed
        self.heading = heading
        self.acceleration = 0
        self.messages = np.zeros(n_messages)
        self.vision_range = vision_range
        self.conspecific_vision_cap = conspecific_vision_cap
        self.heterospecific_vision_cap = heterospecific_vision_cap
        self.species = species
        self.fitness = 0

        self.self_obs_size  = 4
        self.consp_obs_size = 4 + len(self.messages)
        self.hetero_obs_size = 4 
        
        self.full_obs_size =  self.self_obs_size + \
                              self.consp_obs_size * self.conspecific_vision_cap + \
                              self.hetero_obs_size * self.heterospecific_vision_cap
        self.full_act_size = 2 + len(self.messages)

        self.W = np.random.normal(0, 0.05, (self.full_obs_size, self.full_act_size))
        self.b = np.zeros(self.full_act_size)

        self.dt = 0.01

    def step(self):
        neighbors = self.model.space.get_neighbors(self.pos, self.vision_range, include_center=False)
        observations = self.observe(neighbors)
        self.act(observations)

        dx = np.cos(self.heading * pi) * self.speed * self.dt
        dy = np.sin(self.heading * pi) * self.speed * self.dt

        self.pos += dx, dy

    def act(self, observations): 
        acceleration, heading = None, None

        Y = observations @ self.W + self.b

        acceleration, heading = Y[0], Y[1]
        message = np.int(Y[2:])

        self.message = message

        self.speed += acceleration * self.dt
        self.heading = heading
    

    def observe(self, neighbours):
        
        self_obs = np.zeros(self.self_obs_size)
        consp_obs = np.zeros(self.consp_obs_size * self.conspecific_vision_cap)
        hetero_obs = np.zeros(self.hetero_obs_size * self.heterospecific_vision_cap)

        self_obs = np.array(list(self.pos) + [self.heading,  self.speed])

        neighbours = np.array(neighbours)
        neighb_species = [n.species for n in neighbours]
        consp_inds = np.where(neighb_species == self.species)
        hetero_inds = np.where(neighb_species != self.species)

        for i, ind in enumerate(consp_inds):
            if i == self.conspecific_vision_cap:
                break
            consp = neighbours[ind]
            relative_heading = consp.heading - self.heading
            distance = sqrt((self.pos[0] - consp.pos[0])**2 + (self.pos[1] - consp.pos[1])**2)

            this_obs = np.array(list(consp.pos) + [relative_heading, distance] + consp.messages) # size 2 + 2 + n_messages
            consp_obs[i*self.consp_obs_size:(i+1)*self.consp_obs_size] = this_obs

            
        for i, ind in enumerate(hetero_inds):
            if i == self.comm_capacity:
                break
            hetero = neighbours[ind]
            relative_heading = hetero.heading - self.heading
            distance = sqrt((self.pos[0] - hetero.pos[0])**2 + (self.pos[1] - hetero.pos[1])**2)
            this_obs = np.array(list(consp.pos) + [relative_heading, distance])  # size 2 + 2
            hetero_obs[i*self.hetero_obs_size:(i+1)*self.hetero_obs_size] = this_obs

        all_observations = np.hstack((self_obs, consp_obs, hetero_obs))

        return all_observations 


class BoidFlockers(mesa.Model):
    """Flocker model class. Handles agent creation, placement and scheduling."""

    def __init__(
        self,
        seed=None,
        population=100,
        width=100,
        height=100,
        vision=10,
        speed=1,
        separation=1,
        cohere=0.03,
        separate=0.015,
        match=0.05,
        simulator=None,
    ):
        """Create a new Flockers model.

        Args:
            seed: seed for random number generator
            population: Number of Boids
            width: the width of the space
            height: the height of the space
            speed: How fast should the Boids move.
            vision: How far around should each Boid look for its neighbors
            separation: What's the minimum distance each Boid will attempt to keep from any other
            cohere: the relative importance of matching neighbors' positions'
            separate: the relative importance of avoiding close neighbors
            match: factors for the relative importance of
                    the three drives.
            simulator: a Simulator Instance
        """
        super().__init__(seed=seed)
        self.population = population
        self.width = width
        self.height = height
        self.simulator = simulator

        self.schedule = mesa.time.RandomActivation(self)
        self.space = mesa.space.ContinuousSpace(self.width, self.height, True)
        self.factors = {
            "cohere": cohere,
            "separate": separate,
            "match": match,
        }
        self.datacollector = mesa.DataCollector(
            model_reporters={'agent_reporters': {'Pos', 'pos'}}
        )

        for _ in range(self.population):
            x = self.random.random() * self.space.x_max
            y = self.random.random() * self.space.y_max
            pos = np.array((x, y))
            direction = np.random.random(2) * 2 - 1
            boid = Boid(
                model=self,
                speed=speed,
                direction=direction,
                vision=vision,
                separation=separation,
                **self.factors,
            )
            self.space.place_agent(boid, pos)
            self.schedule.add(boid)

    def step(self):
        """Run the model for one step."""
        self.schedule.step()


