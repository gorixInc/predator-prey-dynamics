"""A mesa implementation of our predator and prey agents
"""

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
        initial_weights_disp = 0.1,

        species = 'default'
    ):
        super().__init__(self, model)


        self.speed = 0
        self.max_speed = max_speed
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
        self.alive = True

        self.self_obs_size = 2
        self.consp_obs_size = 4 + len(self.messages)
        self.hetero_obs_size = 4 
        
        self.full_obs_size =  self.self_obs_size + \
                              self.consp_obs_size * self.conspecific_vision_cap + \
                              self.hetero_obs_size * self.heterospecific_vision_cap
        self.full_act_size = 2 + len(self.messages)

        self.W = np.random.normal(0, initial_weights_disp, (self.full_obs_size, self.full_act_size))
        self.b = np.zeros(self.full_act_size)

    def step(self):
        if not self.alive:
            return
        neighbors = self.model.space.get_neighbors(self.pos, self.vision_range, include_center=False)
        observations = self.observe(neighbors)
        self.act(neighbors, observations)

        dx = np.cos(self.heading * pi) * self.speed * self.model.dt
        dy = np.sin(self.heading * pi) * self.speed * self.model.dt
        new_pos = self.pos + np.array([dx, dy])
        self.model.space.move_agent(self, new_pos)
        
    def act(self, neighbours, observations): 
        acceleration, heading = None, None

        Y = observations @ self.W + self.b

        acceleration, heading = Y[0], Y[1]
        heading = np.clip(heading, -1, 1)
        message = Y[2:].astype(int)

        self.message = message
        self.speed += acceleration * self.acceleration_factor * self.model.dt
        self.speed = np.clip(self.speed, -self.max_speed, self.max_speed)
        self.heading = heading
    

    def observe(self, neighbours):
        X_scale = self.model.space.x_max
        Y_scale = self.model.space.y_max

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

            delta_x, delta_y = (self.pos[0] - consp.pos[0])/X_scale, (self.pos[1] - consp.pos[1])/Y_scale
            distance = sqrt(delta_x**2 + delta_y**2)

            this_obs = np.array([delta_x, delta_y] + [relative_heading, distance] + list(consp.messages)) # size 2 + 2 + n_messages
            consp_obs[i*self.consp_obs_size:(i+1)*self.consp_obs_size] = this_obs

            
        for i, ind in enumerate(hetero_inds):
            if i == self.heterospecific_vision_cap:
                break
            hetero = neighbours[ind]
            relative_heading = hetero.heading - self.heading
            
            delta_x, delta_y = (self.pos[0] - hetero.pos[0])/X_scale, (self.pos[1] - hetero.pos[1])/Y_scale
            distance = sqrt(delta_x**2 + delta_y**2)

            this_obs = np.array([delta_x, delta_y] + [relative_heading, distance])  # size 2 + 2
            hetero_obs[i*self.hetero_obs_size:(i+1)*self.hetero_obs_size] = this_obs

        all_observations = np.hstack((self_obs, consp_obs, hetero_obs))
        return all_observations 

class Predator(Animal): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def act(self, neighbours, observations): 
        super().act(neighbours, observations)

        # Eating prey
        neighb_species = np.array([n.species for n in neighbours])
        hetero_inds = np.where(neighb_species != self.species)[0]
        for i, ind in enumerate(hetero_inds):
            hetero = neighbours[ind]
            distance = sqrt((self.pos[0] - hetero.pos[0])**2 + (self.pos[1] - hetero.pos[1])**2)

            if distance < self.size + hetero.size and hetero.alive == True:
                hetero.alive = False
                self.fitness += self.model.predator_catch_reward
                hetero.fitness += self.model.prey_death_reward
                self.model.prey_population -= 1  # decrement prey population

        # Movement fitness punishment
        self.fitness += self.model.predator_movement_reward * abs(self.speed) * self.model.dt

class Prey(Animal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def act(self, neighbours, observations): 
        super().act(neighbours, observations)
        self.fitness += self.model.prey_alive_reward * self.model.dt
        self.fitness += self.model.prey_movement_reward * abs(self.speed) * self.model.dt

class PredatorPreyModel(Model):
    def __init__(self, 
                 predator_population_size=10, 
                 prey_population_size=40,
                 steps_per_generation=500,

                 predator_catch_reward=10,
                 prey_death_reward=-10,
                 predator_movement_reward=-0.1,
                 prey_movement_reward=-0.1,
                 prey_alive_reward=1,

                 prey_max_speed=20,
                 prey_acceleration_factor=5,
                 prey_message_number=0,
                 predator_max_speed=20,
                 predator_acceleration_factor=5,
                 predator_message_number=0,
                 conspecific_vision_cap=4,
                 heterospecific_vision_cap=4,
                 catch_range=1,

                 mutation_rate=0.1,
                 mutation_strength=0.2,
                 mutation_flip_rate=0.01,
                 agent_initial_weights_disp = 0.1,

                 width=100,
                 height=100,
                 dt=0.05,
                ):
        super().__init__()
        self.predator_population_size=predator_population_size
        self.prey_population_size=prey_population_size
        self.steps_per_generation=steps_per_generation

        self.prey_population = prey_population_size
        self.predator_population = prey_population_size

        self.predator_catch_reward=predator_catch_reward
        self.prey_death_reward=prey_death_reward
        self.predator_movement_reward=predator_movement_reward
        self.prey_movement_reward=prey_movement_reward
        self.prey_alive_reward=prey_alive_reward

        self.prey_max_speed=prey_max_speed
        self.prey_acceleration_factor=prey_acceleration_factor
        self.prey_message_number=prey_message_number
        self.predator_max_speed=predator_max_speed
        self.predator_acceleration_factor=predator_acceleration_factor
        self.predator_message_number=predator_message_number
        self.conspecific_vision_cap=conspecific_vision_cap
        self.heterospecific_vision_cap=heterospecific_vision_cap
        self.catch_range=catch_range

        self.mutation_rate=mutation_rate
        self.mutation_strength=mutation_strength
        self.mutation_flip_rate=mutation_flip_rate
        self.agent_initial_weights_disp = agent_initial_weights_disp

        self.width=width
        self.height=height
        self.dt=dt

        self.species_to_class = {'predator': Predator,
                                 'prey': Prey}
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            model_reporters={"Avg_prey_fitness": self.compute_average_prey_fitness,
                             "Avg_predator_fitness": self.compute_average_predator_fitness},
            agent_reporters={"Fitness": "fitness",
                             'Pos': lambda a: (a.pos[0], a.pos[1]),
                             'Speed': 'speed',
                             'Acceleration': 'acceleration',
                             'Message': 'message',
                             'Heading': 'heading',
                             'Species': 'species',
                             'Generation': 'generation',
                             'Alive': 'alive'}
        )
    

        self.space = mesa.space.ContinuousSpace(width, height, True)

        
        self.steps_per_generation = steps_per_generation 
        self.agent_W_shape = None
        self.agent_b_shape = None
        self.step_counter = 0
        self.generation_counter = 0
        self.create_population()

    def compute_average_prey_fitness(self):
        agent_fitnesses = [agent.fitness for agent in self.schedule.agents if agent.species == 'prey']
        return np.mean(agent_fitnesses)
    
    def compute_average_predator_fitness(self):
        agent_fitnesses = [agent.fitness for agent in self.schedule.agents if agent.species == 'predator']
        return np.mean(agent_fitnesses)
    
    def place_agent(self, agent):
        x = self.random.random() * self.space.x_max
        y = self.random.random() * self.space.y_max
        agent.heading = self.random.random() * 2 - 1
        pos = np.array((x, y))
        self.space.place_agent(agent, pos)
        self.schedule.add(agent)
        return agent

    def init_prey(self):
        agent = self.species_to_class['prey'](model=self, species='prey',
                                    max_speed = self.prey_max_speed,
                                    acceleration_factor = self.prey_acceleration_factor,
                                    size = self.catch_range,
                                    vision_range = np.inf,
                                    conspecific_vision_cap = self.conspecific_vision_cap,
                                    heterospecific_vision_cap = self.heterospecific_vision_cap,
                                    n_messages = self.prey_message_number,
                                    initial_weights_disp=self.agent_initial_weights_disp)
        return agent
    def init_predator(self):
        agent = self.species_to_class['predator'](model=self, species='predator',
                                        max_speed = self.predator_max_speed,
                                        acceleration_factor = self.predator_acceleration_factor,
                                        size = self.catch_range,
                                        vision_range = np.inf,
                                        conspecific_vision_cap = self.conspecific_vision_cap,
                                        heterospecific_vision_cap = self.heterospecific_vision_cap,
                                        n_messages = self.predator_message_number,
                                        initial_weights_disp=self.agent_initial_weights_disp)
        return agent

    def create_population(self):
        for i in range(self.predator_population_size):
            agent = self.init_prey()
            agent = self.place_agent(agent)

        for i in range(self.prey_population_size):
            agent = self.init_predator()
            agent = self.place_agent(agent)

        self.agent_W_shape = agent.W.shape
        self.agent_b_shape = agent.b.shape
        
    
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

    def mutate(self, genotype, mutation_rate=0.05, flip_prob=0.005, mutation_strength=0.05):
        """Mutate a genotype by adding Gaussian noise."""
        mutated_genotype = []
        for gene in genotype:
            if random.random() < mutation_rate:
                gene += random.gauss(0, mutation_strength)
            mutated_genotype.append(gene)
            if random.random() < flip_prob:
                gene = -gene
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
            if species == 'predator':
                child1 = self.init_predator()
                child2 = self.init_predator()
            elif species == 'prey':
                child1 = self.init_prey()
                child2 = self.init_prey()
            child1.W, child1.b = child1_W, child1_b
            child2.W, child2.b = child2_W, child2_b
            new_agents.extend([child1, child2])
        if len(new_agents) > n_agents:
            new_agents = new_agents[:n_agents]
        return new_agents


    def step(self):
        """Execute one time step of the model."""

        self.schedule.step()
        self.step_counter += 1
        self.datacollector.collect(self)

        if self.step_counter % self.steps_per_generation == 0 or self.prey_population < 1:
            if self.prey_population < 1:
                print('Ended generation early, all prey dead')
            print(f"Generation {self.generation_counter}")
            self.generation_counter += 1

            # Generate new population using old population
            new_prey = self.create_new_population('prey', self.prey_population_size)
            self.prey_population = len(new_prey)

            new_predators =  self.create_new_population('predator', self.predator_population_size)
            new_agents = new_prey + new_predators

            # Remove old population
            for agent in list(self.schedule.agents):
                self.space.remove_agent(agent)
                self.schedule.remove(agent)
                del agent

            # Add new agents to the schedule and space
            for agent in new_agents:
                self.place_agent(agent)
                agent.generation = self.generation_counter