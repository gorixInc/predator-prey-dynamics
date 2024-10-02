from predator_prey import PredatorPreyModel, Animal

if __name__ == "__main__":
    population_size = 50
    gene_length = 100  # Adjust according to your neural network size
    steps = 1000
    agent_steps_per_gen = 10  # Number of agent steps per generation
    print('asd')
    model = PredatorPreyModel(10, 50, 100)
    for _ in range(steps):
        model.step()

    # Retrieve and plot data
    data = model.datacollector.get_model_vars_dataframe()