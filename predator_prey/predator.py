from predator_prey.animal import Animal
from predator_prey.animal_unicycle import AnimalUnicycle
import numpy as np
from math import sqrt

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

class PredatorUnicicle(AnimalUnicycle):
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