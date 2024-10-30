import numpy as np
import mesa
from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from predator_prey.predator import Predator
from predator_prey.prey import Prey

import random

class PredatorPreyModel(Model):
    def __init__(self, 
                 predator_population_size=10, 
                 prey_population_size=40,
                 steps_per_run=500,
                 runs_per_generation=5,
                 species_to_class=None,

                 predator_catch_reward=10,
                 prey_death_reward=-10,
                 predator_movement_reward=-0.1,
                 predator_reward_sharing_frac=0,
                 predator_reward_sharing_range=0,
                 
                 prey_movement_reward=-0.1,
                 prey_alive_reward=1,

                 prey_max_speed=20,
                 prey_acceleration_factor=5,
                 prey_message_number=0,
                 prey_vision_range=np.inf,
                 prey_hidden_sizes=[],
                 prey_vision_noise=0,

                 predator_max_speed=20,
                 predator_acceleration_factor=5,
                 predator_message_number=0,
                 predator_vision_range=np.inf,
                 predator_hidden_sizes=[],
                 predator_vision_noise=0,
                 
                 conspecific_vision_cap=4,
                 heterospecific_vision_cap=4,
                 catch_range=1,

                 mutation_rate=0.1,
                 mutation_strength=0.2,
                 mutation_flip_rate=0.01,
                 agent_initial_weights_disp = 0.1,

                 mutation_rate_decay=0.98,
                 mutation_strength_decay=0.98,
                 mutation_flip_rate_decay=0.98, 

                 width=100,
                 height=100,
                 dt=0.05,
                 log_every=1,
                ):
        super().__init__()
        self.predator_population_size=predator_population_size
        self.prey_population_size=prey_population_size
        self.steps_per_run=steps_per_run
        self.runs_per_generation = runs_per_generation

        self.species_to_class = species_to_class
        if species_to_class is None:
            self.species_to_class = None
            self.species_to_class = {'predator': Predator,
                                     'prey': Prey}

        self.prey_population = prey_population_size
        self.predator_population = prey_population_size

        self.predator_catch_reward=predator_catch_reward
        self.prey_death_reward=prey_death_reward
        self.predator_reward_sharing_frac = predator_reward_sharing_frac
        self.predator_reward_sharing_range = predator_reward_sharing_range
        self.predator_movement_reward=predator_movement_reward
        self.prey_movement_reward=prey_movement_reward
        self.prey_alive_reward=prey_alive_reward

        self.prey_max_speed=prey_max_speed
        self.prey_acceleration_factor=prey_acceleration_factor
        self.prey_message_number=prey_message_number
        self.prey_vision_range=prey_vision_range
        self.prey_hidden_sizes=prey_hidden_sizes
        self.prey_vision_noise=prey_vision_noise

        self.predator_max_speed=predator_max_speed
        self.predator_acceleration_factor=predator_acceleration_factor
        self.predator_message_number=predator_message_number
        self.predator_vision_range=predator_vision_range
        self.predator_hidden_sizes=predator_hidden_sizes
        self.predator_vision_noise=predator_vision_noise

        self.conspecific_vision_cap=conspecific_vision_cap
        self.heterospecific_vision_cap=heterospecific_vision_cap
        self.catch_range=catch_range

        self.mutation_rate=mutation_rate
        self.mutation_strength=mutation_strength
        self.mutation_flip_rate=mutation_flip_rate
        self.agent_initial_weights_disp = agent_initial_weights_disp
        self.mutation_rate_decay = mutation_rate_decay
        self.mutation_strength_decay = mutation_strength_decay
        self.mutation_flip_rate_decay = mutation_flip_rate_decay

        self.width=width
        self.height=height
        self.dt=dt
        self.log_every=log_every


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
                             'Run': 'run',
                             'Alive': 'alive'}
        )
    

        self.space = mesa.space.ContinuousSpace(width, height, True)

        
        self.steps_per_run = steps_per_run 
        
        self.step_counter = 0
        self.run_steps = 0
        self.generation_steps = 0

        self.generation_counter = 0
        self.run_counter = 0


        self.create_population()

    def compute_average_prey_fitness(self):
        agent_fitnesses = [agent.fitness for agent in self.schedule.agents if agent.species == 'prey']
        return np.mean(agent_fitnesses)
    
    def compute_average_predator_fitness(self):
        agent_fitnesses = [agent.fitness for agent in self.schedule.agents if agent.species == 'predator']
        return np.mean(agent_fitnesses)

    def place_agent(self, agent, move=False):
        x = self.random.random() * self.space.x_max
        y = self.random.random() * self.space.y_max
        agent.heading = self.random.random() * 2 - 1
        pos = np.array((x, y))
        if move:
            self.space.move_agent(agent, pos)
            agent.pos = pos
        else: 
            self.space.place_agent(agent, pos)
            self.schedule.add(agent)
        return agent

    def init_prey(self):
        agent = self.species_to_class['prey'](model=self, species='prey',
                                    max_speed = self.prey_max_speed,
                                    acceleration_factor = self.prey_acceleration_factor,
                                    size = self.catch_range,
                                    vision_range = self.prey_vision_range,
                                    conspecific_vision_cap = self.conspecific_vision_cap,
                                    heterospecific_vision_cap = self.heterospecific_vision_cap,
                                    n_messages = self.prey_message_number,
                                    initial_weights_disp=self.agent_initial_weights_disp,
                                    hidden_sizes=self.prey_hidden_sizes,
                                    vision_noise=self.prey_vision_noise)
        return agent
    def init_predator(self):
        agent = self.species_to_class['predator'](model=self, species='predator',
                                        max_speed = self.predator_max_speed,
                                        acceleration_factor = self.predator_acceleration_factor,
                                        size = self.catch_range,
                                        vision_range = self.predator_vision_range,
                                        conspecific_vision_cap = self.conspecific_vision_cap,
                                        heterospecific_vision_cap = self.heterospecific_vision_cap,
                                        n_messages = self.predator_message_number,
                                        initial_weights_disp=self.agent_initial_weights_disp,
                                        hidden_sizes=self.predator_hidden_sizes,
                                        reward_sharing_range=self.predator_reward_sharing_range,
                                        reward_sharing_frac=self.predator_reward_sharing_frac,
                                        vision_noise=self.predator_vision_noise)
        return agent

    def create_population(self):
        for i in range(self.prey_population_size):
            agent = self.init_prey()
            agent = self.place_agent(agent)

        for i in range(self.predator_population_size):
            agent = self.init_predator()
            agent = self.place_agent(agent)

    
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
        child1_Ws, child2_Ws = [], []
        child1_bs, child2_bs = [], []
        for i in range(len(parent1.Ws)):
            p1w, p2w = parent1.Ws[i], parent2.Ws[i]
            p1b, p2b = parent1.bs[i], parent2.bs[i]

            p1w_f, p2w_f = p1w.flatten(), p2w.flatten()
            c1w_f, c2w_f = self.uniform_crossover(p1w_f, p2w_f)
            c1b, c2b = self.uniform_crossover(p1b, p2b)

            c1w_f, c2w_f = self.mutate(c1w_f), self.mutate(c2w_f)
            c1b, c2b = self.mutate(c1b), self.mutate(c2b)

            c1w, c2w = np.reshape(c1w_f, p1w.shape), np.reshape(c2w_f, p1w.shape)

            child1_Ws.append(c1w)
            child2_Ws.append(c2w)
            child1_bs.append(c1b)
            child2_bs.append(c2b)
        return child1_Ws, child2_Ws, child1_bs, child2_bs

    def mutate(self, genotype):
        """Mutate a genotype by adding Gaussian noise."""
        mutated_genotype = []
        for gene in genotype:
            if random.random() < self.mutation_rate:
                gene += random.gauss(0, self.mutation_strength)
            mutated_genotype.append(gene)
            if random.random() < self.mutation_flip_rate:
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
            child1_Ws, child2_Ws, child1_bs, child2_bs = self.perform_corssover(parent1, parent2)

            # Create new agents
            if species == 'predator':
                child1 = self.init_predator()
                child2 = self.init_predator()
            elif species == 'prey':
                child1 = self.init_prey()
                child2 = self.init_prey()
                
            child1.Ws, child1.bs = child1_Ws, child1_bs
            child2.Ws, child2.bs = child2_Ws, child2_bs
            new_agents.extend([child1, child2])
        if len(new_agents) > n_agents:
            new_agents = new_agents[:n_agents]
        return new_agents


    def step(self):
        """Execute one time step of the model."""

        self.schedule.step()

        self.step_counter += 1
        self.generation_steps += 1
        self.run_steps += 1
        logged = False
        if self.step_counter % self.log_every == 0:
            self.datacollector.collect(self)
            logged = True
        
        if self.run_steps > self.steps_per_run or self.prey_population < 1:
            self.run_steps = 0
            self.run_counter += 1
            print(f'Steps: {self.step_counter}, run: {self.run_counter}, generation: {self.generation_counter}')
            if not logged:
                self.datacollector.collect(self)
                logged = True
            if self.run_counter % self.runs_per_generation == 0:
                print('new generation')
                print(f'Generation {self.generation_counter} run {self.run_counter}')

                self.generation_steps = 0
                self.generation_counter += 1

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

                self.mutation_rate *= self.mutation_rate_decay
                self.mutation_strength *= self.mutation_strength_decay
                self.mutation_flip_rate *= self.mutation_flip_rate_decay

                return

            print('reset_agents')
            self.prey_population = self.prey_population_size
            for agent in list(self.schedule.agents):
                agent.run = self.run_counter % self.runs_per_generation
                agent.reset()
                self.place_agent(agent, move=True)