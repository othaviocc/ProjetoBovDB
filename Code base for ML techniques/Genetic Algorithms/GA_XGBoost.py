from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('dataset.csv', parse_dates=['datetime'])
features = ['EMA_3', 'SMA_3', 'ADXR', 'Bollinger_Norm', 'EMA_5', 'SMA_5', 'std_close3', 'std_open3', 'std_close5', 'std_open5', 'std_open7', 'SMA_7', 'std_close7', 'SMA_9', 'std_open9', 'std_close11', 'EMA_11', 'std_close9']
target = 'trend'  

train_start, train_end = '2024-01-01', '2024-03-30'
val_start, val_end     = '2024-04-01', '2024-06-30'
test_start, test_end   = '2024-07-01', '2024-09-30'

treino = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()
validacao = df[(df['datetime'] >= val_start) & (df['datetime'] <= val_end)].copy()
teste = df[(df['datetime'] >= test_start) & (df['datetime'] <= test_end)].copy()

X_train, y_train_raw = treino[features], treino[target]
X_val, y_val_raw     = validacao[features], validacao[target]
X_test, y_test_raw   = teste[features], teste[target]

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_val = le.transform(y_val_raw)
y_test = le.transform(y_test_raw)

# Configuração do DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# Bits para XGBoost
n_bits_n_estimators = 8     # 50 a 500
n_bits_max_depth = 5        # 3 a 20
n_bits_learning_rate = 8    # 0.001 a 0.3
n_bits_subsample = 8        # 0.5 a 1.0
n_bits_colsample_bytree = 8 # 0.5 a 1.0
n_bits_gamma = 8            # 0 a 5.0 (Regularização)
n_bits_min_child_weight = 4 # 1 a 10

total_bits = (n_bits_n_estimators + n_bits_max_depth + n_bits_learning_rate + 
              n_bits_subsample + n_bits_colsample_bytree + n_bits_gamma + n_bits_min_child_weight)

toolbox.register("attr_bin", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bin, total_bits)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def decode_binary(gene, minimo, maximo, n_bits):
    binary_str = ''.join(map(str, gene))
    int_value = int(binary_str, 2)
    max_value = 2 ** n_bits - 1
    return minimo + (int_value / max_value) * (maximo - minimo)

def evaluate(individual):
    idx = 0
    n_estimators = int(decode_binary(individual[idx:idx+n_bits_n_estimators], 50, 500, n_bits_n_estimators))
    idx += n_bits_n_estimators
    max_depth = int(decode_binary(individual[idx:idx+n_bits_max_depth], 3, 20, n_bits_max_depth))
    idx += n_bits_max_depth
    learning_rate = decode_binary(individual[idx:idx+n_bits_learning_rate], 0.001, 0.3, n_bits_learning_rate)
    idx += n_bits_learning_rate
    subsample = decode_binary(individual[idx:idx+n_bits_subsample], 0.5, 1.0, n_bits_subsample)
    idx += n_bits_subsample
    colsample_bytree = decode_binary(individual[idx:idx+n_bits_colsample_bytree], 0.5, 1.0, n_bits_colsample_bytree)
    idx += n_bits_colsample_bytree
    gamma = decode_binary(individual[idx:idx+n_bits_gamma], 0.0, 5.0, n_bits_gamma)
    idx += n_bits_gamma
    min_child_weight = int(decode_binary(individual[idx:idx+n_bits_min_child_weight], 1, 10, n_bits_min_child_weight))

    model = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=subsample, colsample_bytree=colsample_bytree, gamma=gamma,
        min_child_weight=min_child_weight, eval_metric='logloss', random_state=42, n_jobs=-1
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
    probs = modelo.predict_proba(X)[:, 1] if len(np.unique(y)) == 2 else None

    print(f"\n{'='*15} Métricas para {nome_conjunto} (XGBoost) {'='*15}")
    print("Matriz de Confusão:")
    print(confusion_matrix(y, preds))
    
    print("\nRelatório de Classificação:")
    print(classification_report(y, preds))

    mcc = matthews_corrcoef(y, preds)
    print(f"MCC: {mcc:.4f}")

    if probs is not None:
        fpr, tpr, _ = roc_curve(y, probs)
        roc_auc = auc(fpr, tpr)
        print(f"AUC-ROC: {roc_auc:.4f}")
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Curva ROC XGBoost - {nome_conjunto}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(f'curva_roc_{nome_conjunto.lower()}_xgb.pdf')
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
    pd.DataFrame({
        'Geracao': logbook.select("gen"),
        'Fitness_Minima': logbook.select("min"),
        'Fitness_Media': logbook.select("mean"),
        'Fitness_Maxima': logbook.select("max")
    }).to_csv('historico_fitness_xgb.csv', index=False)

    # Gráfico PDF
    plt.figure(figsize=(10, 6))
    plt.plot(logbook.select("gen"), logbook.select("mean"), label='Média', color='blue', lw=2)
    plt.plot(logbook.select("gen"), logbook.select("max"), label='Máxima', color='green', lw=2)
    plt.xlabel('Gerações')
    plt.ylabel('Fitness (40% Treino + 60% Validação)')
    plt.title('Evolução da Fitness - XGBoost')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.5)
    plt.savefig('grafico_fitness_xgb.pdf')
    plt.close()

    # Decodificar melhor indivíduo
    best_ind = hof[0]
    idx = 0
    n_estimators = int(decode_binary(best_ind[idx:idx+n_bits_n_estimators], 50, 500, n_bits_n_estimators))
    idx += n_bits_n_estimators
    max_depth = int(decode_binary(best_ind[idx:idx+n_bits_max_depth], 3, 20, n_bits_max_depth))
    idx += n_bits_max_depth
    learning_rate = decode_binary(best_ind[idx:idx+n_bits_learning_rate], 0.001, 0.3, n_bits_learning_rate)
    idx += n_bits_learning_rate
    subsample = decode_binary(best_ind[idx:idx+n_bits_subsample], 0.5, 1.0, n_bits_subsample)
    idx += n_bits_subsample
    colsample_bytree = decode_binary(best_ind[idx:idx+n_bits_colsample_bytree], 0.5, 1.0, n_bits_colsample_bytree)
    idx += n_bits_colsample_bytree
    gamma = decode_binary(best_ind[idx:idx+n_bits_gamma], 0.0, 5.0, n_bits_gamma)
    idx += n_bits_gamma
    min_child_weight = int(decode_binary(best_ind[idx:idx+n_bits_min_child_weight], 1, 10, n_bits_min_child_weight))

    print(f'\nMelhor Fitness XGB: {best_ind.fitness.values[0]:.4f}')
    print('Melhores parâmetros encontrados:')
    print(f"  n_estimators     = {n_estimators}")
    print(f"  max_depth        = {max_depth}")
    print(f"  learning_rate    = {learning_rate:.5f}")
    print(f"  subsample        = {subsample:.5f}")
    print(f"  colsample_bytree = {colsample_bytree:.5f}")
    print(f"  gamma            = {gamma:.5f}")
    print(f"  min_child_weight = {min_child_weight}")

    # Treinar modelo final
    model_final = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=subsample, colsample_bytree=colsample_bytree, gamma=gamma,
        min_child_weight=min_child_weight, eval_metric='logloss', random_state=42, n_jobs=-1
    )
    model_final.fit(X_train, y_train)

    calcular_metricas(model_final, X_train, y_train, "TREINO")
    calcular_metricas(model_final, X_test, y_test, "TESTE")

    return best_ind

if __name__ == "__main__":
    best = main()