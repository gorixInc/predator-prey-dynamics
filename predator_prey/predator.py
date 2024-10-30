from predator_prey.animal import Animal
from predator_prey.animal_unicycle import AnimalUnicycle
import numpy as np
from math import sqrt

class Predator(Animal): 
    def __init__(self, reward_sharing_frac=0,
                       reward_sharing_range=0,
                       *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_sharing_frac=reward_sharing_frac
        self.reward_sharing_range=reward_sharing_range


    def act(self, neighbours, observations): 
        super().act(neighbours, observations)

        # Eating prey
        neighb_species = np.array([n.species for n in neighbours])
        hetero_inds = np.where(neighb_species != self.species)[0]
        #consp_inds = 
        for i, ind in enumerate(hetero_inds):
            hetero = neighbours[ind]
            distance = sqrt((self.pos[0] - hetero.pos[0])**2 + (self.pos[1] - hetero.pos[1])**2)

            if distance < self.size + hetero.size and hetero.alive == True:
                hetero.alive = False
                self.fitness += self.model.predator_catch_reward
                # Reward sharing
                if self.reward_sharing_range > 0:
                    sharing_neighbs = self.model.space.get_neighbors(self.pos, 
                                                             self.reward_sharing_range, 
                                                             include_center=False)
                    sharing_species = np.array([n.species for n in sharing_neighbs])
                    sharing_consp_inds = np.where(sharing_species == self.species)[0]
                    for sh_ind in sharing_consp_inds:
                        sh_consp = sharing_neighbs[sh_ind]
                        sh_consp.fitness += self.model.predator_catch_reward * self.reward_sharing_frac

                hetero.fitness += self.model.prey_death_reward
                self.model.prey_population -= 1  # decrement prey population

        # Movement fitness punishment
        self.fitness += self.model.predator_movement_reward * abs(self.speed) * self.model.dt

class PredatorUnicicle(AnimalUnicycle):
    def __init__(self, reward_sharing_frac=0,
                       reward_sharing_range=0,
                       *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_sharing_frac=reward_sharing_frac
        self.reward_sharing_range=reward_sharing_range

    def act(self, neighbours, observations): 
        super().act(neighbours, observations)

        # Eating prey
        neighb_species = np.array([n.species for n in neighbours])
        hetero_inds = np.where(neighb_species != self.species)[0]
        #consp_inds = 
        for i, ind in enumerate(hetero_inds):
            hetero = neighbours[ind]
            distance = sqrt((self.pos[0] - hetero.pos[0])**2 + (self.pos[1] - hetero.pos[1])**2)

            if distance < self.size + hetero.size and hetero.alive == True:
                hetero.alive = False
                self.fitness += self.model.predator_catch_reward
                # Reward sharing
                if self.reward_sharing_range > 0:
                    sharing_neighbs = self.model.space.get_neighbors(self.pos, 
                                                             self.reward_sharing_range, 
                                                             include_center=False)
                    sharing_species = np.array([n.species for n in sharing_neighbs])
                    sharing_consp_inds = np.where(sharing_species == self.species)[0]
                    for sh_ind in sharing_consp_inds:
                        sh_consp = sharing_neighbs[sh_ind]
                        
                        sh_consp.fitness += self.model.predator_catch_reward * self.reward_sharing_frac

                hetero.fitness += self.model.prey_death_reward
                self.model.prey_population -= 1  # decrement prey population

        # Movement fitness punishment
        self.fitness += self.model.predator_movement_reward * abs(self.speed) * self.model.dt