from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('data_training.csv', parse_dates=['datetime'])
'''
features = ['NSMA_5', 'NSMA_7', 'Bands_Norm'] 
features = ['Bands_Norm', 'NSMA_3', 'NSMA_5']
'''
features = ['Bands_Norm', 'NSMA_3', 'NSMA_5', 'NSMA_7', 'NSMA_9']

target = 'trend'  

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

param_ranges = {
    'hidden_layer_sizes': (10, 200),           
    'activation': (0, 2),                        
    'alpha': (0.0001, 0.1),                    
    'learning_rate_init': (0.0001, 0.1),        
    'max_iter': (100, 500),                      
}

param_bits = {
    'hidden_layer_sizes': 8,
    'activation': 2,
    'alpha': 10,
    'learning_rate_init': 10,
    'max_iter': 9,
}

activation_map = {
    0: 'relu',
    1: 'tanh',
    2: 'logistic'
}

total_bits = sum(param_bits.values())

#deap criação
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

toolbox.register("attr_bin", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bin, total_bits)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def decode_binary(gene, minimo, maximo, n_bits):
    binary_str = ''.join(map(str, gene))
    int_value = int(binary_str, 2)
    max_value = 2 ** n_bits - 1
    return minimo + (int_value / max_value) * (maximo - minimo)

def decode_individual(individual):
    idx = 0
    decoded = {}
    for param, (min_val, max_val) in param_ranges.items():
        bits = param_bits[param]
        value = decode_binary(individual[idx:idx+bits], min_val, max_val, bits)
        if param in ['hidden_layer_sizes', 'max_iter']:
            value = int(round(value))
        elif param == 'activation':
            value = activation_map[int(round(value))]
        decoded[param] = value
        idx += bits
    return decoded

def calculate_fitness(individual):
    params = decode_individual(individual)
    model = MLPClassifier(
        hidden_layer_sizes=(params['hidden_layer_sizes'],),
        activation=params['activation'],
        alpha=params['alpha'],
        learning_rate_init=params['learning_rate_init'],
        max_iter=params['max_iter'],
        random_state=42
    )
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
    except:
        acc = 0.0
    return (acc,)

toolbox.register("evaluate", calculate_fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main(n_gen=1000, pop_size=20):
    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("mean", np.mean)
    stats.register("max", np.max)

    algorithms.eaSimple(population, 
                        toolbox, 
                        cxpb=0.8, 
                        mutpb=0.05, 
                        ngen=n_gen, 
                        stats=stats, 
                        halloffame=hof, 
                        verbose=True)

    best_individual = hof[0]
    best_params = decode_individual(best_individual)
    best_acc = best_individual.fitness.values[0]

    print('\nMelhor indivíduo (acurácia: {:.4f}):'.format(best_acc))
    for k, v in best_params.items():
        print(f"{k}: {v}")

    return best_individual, best_params, best_acc

if __name__ == '__main__':
    main()
