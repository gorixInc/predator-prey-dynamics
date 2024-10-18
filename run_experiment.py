from predator_prey.model import PredatorPreyModel
from predator_prey.prey import  Prey, PreyUnicicle
from predator_prey.predator import Predator, PredatorUnicicle
from experiment_params import simple_params, unicycle_params
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import argparse
from scipy.spatial import cKDTree

import pickle
import os


def load_yaml_to_dict(file_path):
    """Loads a YAML file and returns its contents as a dictionary."""
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Read a YAML file and convert it to a dictionary.')
    parser.add_argument('yaml_path', type=str, help='Path to the YAML file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load the YAML file into a dictionary
    params = load_yaml_to_dict(args.yaml_path)
    general_params = params['general_params']
    model_params = params['model_params']

    # Print the dictionary for demonstration purposes
    movement_type = model_params.pop('movement_type')
    if movement_type == 'simple':
        model_params['species_to_class'] = {
        'predator': Predator,
        'prey': Prey
    }
    elif movement_type == 'unicycle': 
        model_params['species_to_class'] = {
            'predator': PredatorUnicicle,
            'prey': PredatorUnicicle
        }

generations = general_params['generations']
model = PredatorPreyModel(**model_params)   

while model.generation_counter < generations:
    model.step()

agent_data = model.datacollector.get_agent_vars_dataframe()
model_data = model.datacollector.get_model_vars_dataframe()


experiment_name = general_params['experiment_name']
experiment_dir = f'experiments/{experiment_name}'
agent_data.index = agent_data.index.set_levels(
    agent_data.index.levels[1].astype(str),
    level=1,
)
os.makedirs(experiment_dir, exist_ok=True)
agent_data.to_hdf(f'{experiment_dir}/agent_data.hdf', key='agent_data', mode='w')
model_data.to_hdf(f'{experiment_dir}/model_data.hdf', key='model_data', mode='w')
pickle.dump(model_params, open(f'{experiment_dir}/params.pkl', "wb"))

generations = agent_data['Generation'].unique()
def get_mean_nnd(data):
    mean_nnd_over_time = []
    time_steps = data.index.get_level_values('Step').unique()
# Loop over time steps
    for step in time_steps:
        # Extract positions
        agent_positions_step = data.xs(step, level='Step').copy()
        positions = agent_positions_step['Pos'].tolist()
        positions_array = np.array(positions)
        
        # Calculate NND
        if len(positions_array) >= 2:
            tree = cKDTree(positions_array)
            distances, indices = tree.query(positions_array, k=2)
            nearest_distances = distances[:, 1]
            mean_nnd = np.mean(nearest_distances)
        else:
            nearest_distances = np.array([])
            mean_nnd = np.nan
        mean_nnd_over_time.append(mean_nnd)
    return np.mean(mean_nnd_over_time)


def get_avg_remaining_prey(data):
    runs = data['Run'].unique()
    alive_prey_per_run = []
    for run in runs: 
        run_data = data[data['Run'] == run]
        max_step = run_data.index.get_level_values(0).max()
        last_step = run_data.xs(max_step, 0)
        last_step_prey = last_step[last_step['Species'] == 'prey']
        n_alive_prey = np.sum(last_step_prey['Alive'])
        alive_prey_per_run.append(n_alive_prey)
    return np.mean(alive_prey_per_run)

prey_nnd = []
predator_nnd = []
average_alive_prey = []
for generation in generations:
    generation_data = agent_data[agent_data['Generation'] == generation]
    prey_data = generation_data[generation_data['Species'] == 'prey']
    predator_data = generation_data[generation_data['Species'] == 'predator']
    prey_nnd.append(get_mean_nnd(prey_data))
    predator_nnd.append(get_mean_nnd(predator_data))
    average_alive_prey.append(get_avg_remaining_prey(generation_data))


generation_data_df = pd.DataFrame(data={
  'generation': generations,
  'predator_nnd': predator_nnd,
  'prey_nnd': prey_nnd,
  'predator_nnd': predator_nnd
})

generation_data_df.to_csv(f'{experiment_dir}/generation_metrics.csv')

plt.plot(model_data.index, model_data.Avg_prey_fitness, label='prey')
plt.plot(model_data.index, model_data.Avg_predator_fitness, label='predator')
plt.legend()
plt.xlabel('Step number')
plt.ylabel('Fitness')
plt.tight_layout()
plt.savefig(f'{experiment_dir}/fitness.png')

plt.figure(figsize=(10, 6))
plt.plot(generations, prey_nnd, marker='o', label='prey')
plt.plot(generations, predator_nnd, marker='o', label='predator')
plt.xlabel('Generation')
plt.ylabel('Mean Nearest Neighbor Distance')
plt.title('Mean NND Over Generation')
plt.legend()
plt.grid(True)
plt.savefig(f'{experiment_dir}/nnd.png')


plt.figure(figsize=(10, 6))
plt.plot(generations, average_alive_prey, marker='o', label='prey')
plt.xlabel('Generation')
plt.ylabel('Number of Prey')
plt.title('Average number of Prey left at the end of run per generation')
plt.legend()
plt.grid(True)
plt.savefig(f'{experiment_dir}/alive_prey.png')