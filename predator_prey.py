"""A mesa implementation of our predator and prey agents
"""

import numpy as np
from math import sqrt, pi, cos, sin
import mesa
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
import matplotlib.pyplot as plt
from mesa.time import BaseScheduler


class Animal(Agent):
    def __init__(
        self,
        model,
        speed = 0,
        heading = 0,
        vision_range = np.inf,
        conspecific_vision_cap = 4,
        heterospecific_vision_cap = 4,
        fitness_speed_decay = 0,
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
        self.fitness_speed_decay = fitness_speed_decay

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

        self.pos += (dx, dy)

    def act(self, observations): 
        acceleration, heading = None, None

        Y = observations @ self.W + self.b

        acceleration, heading = Y[0], Y[1]
        message = Y[2:].astype(int)

        self.message = message

        self.speed += acceleration * self.dt
        self.heading = heading
    

    def observe(self, neighbours):
        
        self_obs = np.zeros(self.self_obs_size)
        consp_obs = np.zeros(self.consp_obs_size * self.conspecific_vision_cap)
        hetero_obs = np.zeros(self.hetero_obs_size * self.heterospecific_vision_cap)

        self_obs = np.array(list(self.pos) + [self.heading,  self.speed])

        neighbours = np.array(neighbours)
        neighb_species = np.array([n.species for n in neighbours])
        consp_inds = np.where(neighb_species == self.species)[0]
        hetero_inds = np.where(neighb_species != self.species)[0]

        for i, ind in enumerate(consp_inds):
            if i == self.conspecific_vision_cap:
                break
            consp = neighbours[ind]
            relative_heading = consp.heading - self.heading
            distance = sqrt((self.pos[0] - consp.pos[0])**2 + (self.pos[1] - consp.pos[1])**2)

            this_obs = np.array(list(consp.pos) + [relative_heading, distance] + list(consp.messages)) # size 2 + 2 + n_messages
            consp_obs[i*self.consp_obs_size:(i+1)*self.consp_obs_size] = this_obs

            
        for i, ind in enumerate(hetero_inds):
            if i == self.heterospecific_vision_cap:
                break
            hetero = neighbours[ind]
            relative_heading = hetero.heading - self.heading
            distance = sqrt((self.pos[0] - hetero.pos[0])**2 + (self.pos[1] - hetero.pos[1])**2)
            this_obs = np.array(list(consp.pos) + [relative_heading, distance])  # size 2 + 2
            hetero_obs[i*self.hetero_obs_size:(i+1)*self.hetero_obs_size] = this_obs

        all_observations = np.hstack((self_obs, consp_obs, hetero_obs))

        return all_observations 


class PredatorPreyModel(Model):
    def __init__(self, 
                 predator_population_size, 
                 prey_population_size,
                 steps_per_generation,
                 width = 100,
                 height = 100
                ):
        self.predator_population_size = predator_population_size
        self.prey_population_size = prey_population_size

        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            model_reporters={'agent_reporters': {'Pos', 'pos'}}
        )

        self.space = mesa.space.ContinuousSpace(width, height, True)

        
        self.steps_per_generation = steps_per_generation 
        self.agent_W_shape = None
        self.agent_b_shape = None
        self.step_counter = 0
        self.create_population()

    def init_agent(self, species):
        x = self.random.random() * self.space.x_max
        y = self.random.random() * self.space.y_max
        pos = np.array((x, y))
        agent = Animal(model=self, species=species)
        self.space.place_agent(agent, pos)
        self.schedule.add(agent)
        return agent

    def create_population(self):
        for i in range(self.predator_population_size):
            agent = self.init_agent('predator')

        for i in range(self.prey_population_size):
            agent = self.init_agent('prey')

        self.agent_W_shape = agent.W.shape
        self.agent_b_shape = agent.b.shape
        
        
    def adjust_fitnesses(self):
        """Adjust fitness values to be positive for roulette wheel selection."""
        min_fitness = min(agent.fitness for agent in self.schedule.agents)
        shift = -min_fitness + 1e-6 if min_fitness < 0 else 1e-6
        for agent in self.schedule.agents:
            agent.adjusted_fitness = agent.fitness + shift

    def roulette_wheel_selection(self, species):
        """Select an agent based on roulette wheel selection."""
        all_fitnesses = [agent.fitness for agent in self.schedule.agents if agent.species == species]
        min_fitness = min(all_fitnesses)
        shift = -min_fitness + 1e-6 if min_fitness < 0 else 1e-6
        adj_fitnesses = [agent.fitness + shift for agent in self.schedule.agents if agent.species == species]
        total_fitness = sum(adj_fitnesses)

        pick = random.uniform(0, total_fitness)
        current = 0
        for agent in self.schedule.agents:
            if not agent.species == species:
                continue
            current += agent.fitness + shift
            if current > pick:
                return agent
        # In case of rounding errors
        return self.schedule.agents[-1]
    
    def uniform_crossover(self, genotype1, genotype2):
        child1_genotype = []
        child2_genotype = []
        for gene1, gene2 in zip(genotype1, genotype2):
            if random.random() < 0.5:
                child1_genotype.append(gene1)
                child2_genotype.append(gene2)
            else:
                child1_genotype.append(gene2)
                child2_genotype.append(gene1)
        return np.array(child1_genotype), np.array(child2_genotype)
    
    def perform_corssover(self, parent1, parent2): 
        child1_W, child2_W = None, None
        child1_b, child2_b = None, None

        parent1_W, parent2_W = parent1.W.flatten(), parent2.W.flatten()
        parent1_b, parent2_b = parent1.b, parent2.b

        child1_W, child2_W = self.uniform_crossover(parent1_W, parent2_W)
        child1_b, child2_b = self.uniform_crossover(parent1_b, parent2_b)

        child1_W, child2_W = self.mutate(child1_W), self.mutate(child2_W)
        child1_b, child2_b = self.mutate(child1_b), self.mutate(child2_b)

        child1_W, child2_W = np.reshape(child1_W, self.agent_W_shape), np.reshape(child2_W, self.agent_W_shape)
        return child1_W, child1_b, child2_W, child2_b

    def mutate(self, genotype, mutation_rate=0.1, mutation_strength=0.05):
        """Mutate a genotype by adding Gaussian noise."""
        mutated_genotype = []
        for gene in genotype:
            if random.random() < mutation_rate:
                gene += random.gauss(0, mutation_strength)
            mutated_genotype.append(gene)
        return np.array(mutated_genotype)
    
    def create_new_population(self, species, n_agents):
        new_agents = []
        # Generate new population
        while len(new_agents) < n_agents:
            parent1 = self.roulette_wheel_selection(species)
            parent2 = self.roulette_wheel_selection(species)

            # Ensure parents are not the same agent
            while parent2 == parent1:
                parent2 = self.roulette_wheel_selection(species)

            # Crossover
            child1_W, child1_b, child2_W, child2_b = self.perform_corssover(parent1, parent2)

            # Create new agents
            child1 = self.init_agent(species)
            child2 = self.init_agent(species)
            child1.W, child1.b = child1_W, child1_b
            child2.W, child2.b = child2_W, child2_b
            new_agents.extend([child1, child2])
        if len(new_agents) > n_agents:
            new_agents = new_agents[:n_agents]
        return new_agents


    def step(self):
        """Execute one time step of the model."""

        self.schedule.step()
        print(self.step_counter)
        self.step_counter += 1

        # Adjust fitnesses for selection


        # Collect data
        #self.datacollector.collect(self)
        if self.step_counter % self.steps_per_generation == 0:
            self.adjust_fitnesses()
            new_agents = []
            # Generate new population
            new_prey = self.create_new_population('prey', self.prey_population_size)
            new_predators =  self.create_new_population('predator', self.prey_population_size)
            new_agents = new_prey + new_predators
            # Update the schedule with new agents
            for agent in list(self.schedule.agents):
                self.schedule.remove(agent)

            # Add new agents to the schedule
            for agent in new_agents:
                self.schedule.add(agent)
