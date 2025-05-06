from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carrega os dados
df = pd.read_csv('data_training.csv', parse_dates=['datetime'])

features = ['NSMA_5', 'NSMA_7', 'Bands_Norm']

'''
features = ['Bands_Norm', 'NSMA_3', 'NSMA_5', 'NSMA_7', 'NSMA_9']   #Relief

features = ['Bands_Norm', 'NSMA_3', 'NSMA_5']
                                                      #esses sao outros exemplos que sao necessarios fazer.
features = ['NSMA_5', 'NSMA_7', 'Bands_Norm'] 
'''

target = 'trend'  

train_start, train_end = '2024-01-01', '2024-03-30'
test_start, test_end = '2024-04-01', '2024-06-30'

treino = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()
validacao = df[(df['datetime'] >= test_start) & (df['datetime'] <= test_end)].copy()

X_train = treino[features]
y_train = treino[target]
X_test = validacao[features]
y_test = validacao[target]

# Configuração AG
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

n_bits_estimators = 8       # 10–200
n_bits_max_depth = 5          # 1–30
n_bits_max_features = 3           # 1–5
n_bits_min_samples_leaf = 4       # 1–20
n_bits_min_samples_split = 4            # 2–20

total_bits = n_bits_estimators + n_bits_max_depth + n_bits_max_features + n_bits_min_samples_leaf + n_bits_min_samples_split

toolbox.register("attr_bin", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bin, total_bits)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def decode_binary(gene, minimo, maximo, n_bits):
    binary_str = ''.join(map(str, gene))
    int_value = int(binary_str, 2)
    max_value = 2 ** n_bits - 1
    return int(minimo + (int_value / max_value) * (maximo - minimo))

def calculate_fitness(params):
    model = RandomForestClassifier(
        n_estimators=params[0],
        max_depth=params[1],
        max_features=params[2],
        min_samples_leaf=params[3],
        min_samples_split=params[4],
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc,

def evaluate(individual):
    idx = 0
    est = decode_binary(individual[idx:idx + n_bits_estimators], 10, 200, n_bits_estimators)
    idx += n_bits_estimators
    depth = decode_binary(individual[idx:idx + n_bits_max_depth], 1, 30, n_bits_max_depth)
    idx += n_bits_max_depth
    max_feat = decode_binary(individual[idx:idx + n_bits_max_features], 1, len(features), n_bits_max_features)
    idx += n_bits_max_features
    leaf = decode_binary(individual[idx:idx + n_bits_min_samples_leaf], 2, 20, n_bits_min_samples_leaf)
    idx += n_bits_min_samples_leaf
    split = decode_binary(individual[idx:idx + n_bits_min_samples_split], 2, 20, n_bits_min_samples_split)

    return calculate_fitness([est, depth, max_feat, leaf, split])

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main(n_gen=100, pop_size=20):    #population arrumar
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
    idx = 0
    est = decode_binary(best_individual[idx:idx + n_bits_estimators], 10, 200, n_bits_estimators)
    idx += n_bits_estimators
    depth = decode_binary(best_individual[idx:idx + n_bits_max_depth], 1, 30, n_bits_max_depth)
    idx += n_bits_max_depth
    max_feat = decode_binary(best_individual[idx:idx + n_bits_max_features], 1, len(features), n_bits_max_features)
    idx += n_bits_max_features
    leaf = decode_binary(best_individual[idx:idx + n_bits_min_samples_leaf], 1, 20, n_bits_min_samples_leaf)
    idx += n_bits_min_samples_leaf
    split = decode_binary(best_individual[idx:idx + n_bits_min_samples_split], 2, 20, n_bits_min_samples_split)

    print(f'\nMelhor acurácia: {best_individual.fitness.values[0]:.4f}')
    print(f'Melhores parâmetros encontrados:')
    print(f'  n_estimators     = {est}')
    print(f'  max_depth        = {depth}')
    print(f'  max_features     = {max_feat}')
    print(f'  min_samples_leaf = {leaf}')
    print(f'  min_samples_split= {split}')

    return best_individual

best = main()