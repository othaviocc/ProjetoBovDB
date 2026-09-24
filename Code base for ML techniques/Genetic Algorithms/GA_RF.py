from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, matthews_corrcoef
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('dataset.csv', parse_dates=['datetime'])

features = ['EMA_3', 'SMA_3', 'ADXR', 'Bollinger_Norm', 'EMA_5', 'SMA_5', 'std_close3', 'std_open3', 'std_close5', 'std_open5', 'std_open7', 'SMA_7', 'std_close7', 'SMA_9', 'std_open9', 'std_close11', 'EMA_11', 'std_close9']
target = 'trend'  

# Divisão Temporal (Treino, Validação, Teste)
train_start, train_end = '2024-01-01', '2024-03-30'
val_start, val_end     = '2024-04-01', '2024-06-30'
test_start, test_end   = '2024-07-01', '2024-09-30'

treino = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()
validacao = df[(df['datetime'] >= val_start) & (df['datetime'] <= val_end)].copy()
teste = df[(df['datetime'] >= test_start) & (df['datetime'] <= test_end)].copy()

X_train, y_train = treino[features], treino[target]
X_val, y_val     = validacao[features], validacao[target]
X_test, y_test   = teste[features], teste[target]

# Configuração do DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# Bits para Random Forest
n_bits_n_estimators = 8     # 50 a 500 árvores
n_bits_max_depth = 6        # 3 a 50 de profundidade
n_bits_min_samples_split = 5 # 2 a 20 amostras
n_bits_min_samples_leaf = 5  # 1 a 20 amostras
n_bits_max_features = 2     # 0: 'sqrt', 1: 'log2', 2: None
n_bits_criterion = 1        # 0: 'gini', 1: 'entropy'

total_bits = (n_bits_n_estimators + n_bits_max_depth + n_bits_min_samples_split + 
              n_bits_min_samples_leaf + n_bits_max_features + n_bits_criterion)

toolbox.register("attr_bin", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bin, total_bits)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def decode_binary(gene, minimo, maximo, n_bits):
    if n_bits == 0: return minimo
    binary_str = ''.join(map(str, gene))
    int_value = int(binary_str, 2)
    max_value = 2 ** n_bits - 1
    return minimo + (int_value / max_value) * (maximo - minimo)

def evaluate(individual):
    idx = 0
    n_estimators = int(decode_binary(individual[idx:idx+n_bits_n_estimators], 50, 500, n_bits_n_estimators))
    idx += n_bits_n_estimators

    max_depth = int(decode_binary(individual[idx:idx+n_bits_max_depth], 3, 50, n_bits_max_depth))
    idx += n_bits_max_depth

    min_samples_split = int(decode_binary(individual[idx:idx+n_bits_min_samples_split], 2, 20, n_bits_min_samples_split))
    idx += n_bits_min_samples_split

    min_samples_leaf = int(decode_binary(individual[idx:idx+n_bits_min_samples_leaf], 1, 20, n_bits_min_samples_leaf))
    idx += n_bits_min_samples_leaf

    max_features_opts = ['sqrt', 'log2', None, 'sqrt']
    max_features_idx = int(decode_binary(individual[idx:idx+n_bits_max_features], 0, 3, n_bits_max_features))
    max_features = max_features_opts[max_features_idx]
    idx += n_bits_max_features

    criterion = 'entropy' if individual[idx] == 1 else 'gini'

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        random_state=42,
        n_jobs=-1 # Paralelização para acelerar
    )

    try:
        model.fit(X_train, y_train)
        acc_train = accuracy_score(y_train, model.predict(X_train))
        acc_val = accuracy_score(y_val, model.predict(X_val))
        fitness = (0.4 * acc_train) + (0.6 * acc_val)
    except:
        fitness = 0.0 

    return fitness,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def calcular_metricas(modelo, X, y, nome_conjunto):
    preds = modelo.predict(X)
    probs = modelo.predict_proba(X)[:, 1] if len(modelo.classes_) == 2 else None

    print(f"\n{'='*15} Métricas para {nome_conjunto} (RF) {'='*15}")
    print("Matriz de Confusão:")
    print(confusion_matrix(y, preds))
    
    print("\nRelatório de Classificação:")
    print(classification_report(y, preds))

    mcc = matthews_corrcoef(y, preds)
    print(f"MCC: {mcc:.4f}")

    if probs is not None:
        fpr, tpr, _ = roc_curve(y, probs, pos_label=modelo.classes_[1])
        roc_auc = auc(fpr, tpr)
        print(f"AUC-ROC: {roc_auc:.4f}")
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Curva ROC RF - {nome_conjunto}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(f'curva_roc_{nome_conjunto.lower()}_rf.pdf')
        plt.close()

def main(n_gen=30, pop_size=30):
    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("mean", np.mean)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.8, mutpb=0.05, ngen=n_gen,
                                       stats=stats, halloffame=hof, verbose=True)

    # Exportar histórico
    historico_df = pd.DataFrame({
        'Geracao': logbook.select("gen"),
        'Fitness_Minima': logbook.select("min"),
        'Fitness_Media': logbook.select("mean"),
        'Fitness_Maxima': logbook.select("max")
    })
    historico_df.to_csv('historico_fitness_rf.csv', index=False)

    # Gráfico PDF
    plt.figure(figsize=(10, 6))
    plt.plot(logbook.select("gen"), logbook.select("mean"), label='Média', color='blue', lw=2)
    plt.plot(logbook.select("gen"), logbook.select("max"), label='Máxima', color='green', lw=2)
    plt.xlabel('Gerações')
    plt.ylabel('Fitness (40% Treino + 60% Validação)')
    plt.title('Evolução da Fitness - Random Forest')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.5)
    plt.savefig('grafico_fitness_rf.pdf')
    plt.close()

    # Decodificar melhor indivíduo
    best_ind = hof[0]
    idx = 0
    n_estimators = int(decode_binary(best_ind[idx:idx+n_bits_n_estimators], 50, 500, n_bits_n_estimators))
    idx += n_bits_n_estimators
    max_depth = int(decode_binary(best_ind[idx:idx+n_bits_max_depth], 3, 50, n_bits_max_depth))
    idx += n_bits_max_depth
    min_samples_split = int(decode_binary(best_ind[idx:idx+n_bits_min_samples_split], 2, 20, n_bits_min_samples_split))
    idx += n_bits_min_samples_split
    min_samples_leaf = int(decode_binary(best_ind[idx:idx+n_bits_min_samples_leaf], 1, 20, n_bits_min_samples_leaf))
    idx += n_bits_min_samples_leaf
    max_features_opts = ['sqrt', 'log2', None, 'sqrt']
    max_features = max_features_opts[int(decode_binary(best_ind[idx:idx+n_bits_max_features], 0, 3, n_bits_max_features))]
    idx += n_bits_max_features
    criterion = 'entropy' if best_ind[idx] == 1 else 'gini'

    print(f'\nMelhor Fitness RF: {best_ind.fitness.values[0]:.4f}')
    print('Melhores parâmetros encontrados:')
    print(f"  n_estimators      = {n_estimators}")
    print(f"  max_depth         = {max_depth}")
    print(f"  min_samples_split = {min_samples_split}")
    print(f"  min_samples_leaf  = {min_samples_leaf}")
    print(f"  max_features      = {max_features}")
    print(f"  criterion         = {criterion}")

    # Treinar modelo final
    model_final = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        max_features=max_features, criterion=criterion, random_state=42, n_jobs=-1
    )
    model_final.fit(X_train, y_train)

    calcular_metricas(model_final, X_train, y_train, "TREINO")
    calcular_metricas(model_final, X_test, y_test, "TESTE")

    return best_ind

if __name__ == "__main__":
    best = main()