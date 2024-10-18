from predator_prey.animal import Animal
from predator_prey.animal_unicycle import AnimalUnicycle

class Prey(Animal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def act(self, neighbours, observations): 
        super().act(neighbours, observations)
        self.fitness += self.model.prey_alive_reward * self.model.dt
        self.fitness += self.model.prey_movement_reward * abs(self.speed) * self.model.dt

class PreyUnicicle(AnimalUnicycle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def act(self, neighbours, observations): 
        super().act(neighbours, observations)
        self.fitness += self.model.prey_alive_reward * self.model.dt
        self.fitness += self.model.prey_movement_reward * abs(self.speed) * self.model.dt
