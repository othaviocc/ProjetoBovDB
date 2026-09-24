from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Carregamento dos dados
df = pd.read_csv('dataset.csv', parse_dates=['datetime'])

features = ['EMA_3', 'SMA_3', 'ADXR', 'Bollinger_Norm', 'EMA_5', 'SMA_5', 'std_close3', 'std_open3', 'std_close5', 'std_open5', 'std_open7', 'SMA_7', 'std_close7', 'SMA_9', 'std_open9', 'std_close11', 'EMA_11', 'std_close9']
target = 'trend'  

train_start, train_end = '2024-01-01', '2024-03-30'
val_start, val_end     = '2024-04-01', '2024-06-30'
test_start, test_end   = '2024-07-01', '2024-09-30' 

treino = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()
validacao = df[(df['datetime'] >= val_start) & (df['datetime'] <= val_end)].copy()
teste = df[(df['datetime'] >= test_start) & (df['datetime'] <= test_end)].copy()

X_train, y_train = treino[features], treino[target]
X_val, y_val     = validacao[features], validacao[target]
X_test, y_test   = teste[features], teste[target]

# Escalonamento
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Configuração do DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

n_bits_hidden_layer1 = 5    
n_bits_hidden_layer2 = 5    
n_bits_activation    = 2       
n_bits_solver        = 2           
n_bits_alpha         = 8            
n_bits_lr_init       = 8          

total_bits = (n_bits_hidden_layer1 + n_bits_hidden_layer2 + n_bits_activation +
              n_bits_solver + n_bits_alpha + n_bits_lr_init)

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
    hidden1 = int(decode_binary(individual[idx:idx+n_bits_hidden_layer1], 1, 32, n_bits_hidden_layer1))
    idx += n_bits_hidden_layer1

    hidden2 = int(decode_binary(individual[idx:idx+n_bits_hidden_layer2], 0, 32, n_bits_hidden_layer2))
    idx += n_bits_hidden_layer2

    activations = ['identity', 'logistic', 'tanh', 'relu']
    solvers = ['lbfgs', 'sgd', 'adam']

    activation_idx = min(int(decode_binary(individual[idx:idx+n_bits_activation], 0, len(activations)-1, n_bits_activation)), len(activations)-1)
    idx += n_bits_activation

    solver_idx = min(int(decode_binary(individual[idx:idx+n_bits_solver], 0, len(solvers)-1, n_bits_solver)), len(solvers)-1)
    idx += n_bits_solver

    alpha = decode_binary(individual[idx:idx+n_bits_alpha], 0.0001, 0.1, n_bits_alpha)
    idx += n_bits_alpha

    lr_init = decode_binary(individual[idx:idx+n_bits_lr_init], 0.0001, 0.1, n_bits_lr_init)

    hidden_layers = (hidden1,) if hidden2 == 0 else (hidden1, hidden2)

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activations[activation_idx],
        solver=solvers[solver_idx],
        alpha=alpha,
        learning_rate_init=lr_init,
        max_iter=500,  
        random_state=42
    )

    try:
        model.fit(X_train, y_train)        
        preds_train = model.predict(X_train)
        preds_val = model.predict(X_val)
        
        acc_train = accuracy_score(y_train, preds_train)
        acc_val = accuracy_score(y_val, preds_val)
        
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
    
    # Verifica se as classes são binárias para calcular AUC-ROC corretamente
    probs = modelo.predict_proba(X)[:, 1] if len(modelo.classes_) == 2 else None

    print(f"\n{'='*15} Métricas para {nome_conjunto} {'='*15}")
    print("Matriz de Confusão:")
    print(confusion_matrix(y, preds))
    
    print("\nRelatório de Classificação (Precision, Recall, F1-Score):")
    print(classification_report(y, preds))

    mcc = matthews_corrcoef(y, preds)
    print(f"MCC (Matthews Correlation Coefficient): {mcc:.4f}")

    if probs is not None:
        fpr, tpr, _ = roc_curve(y, probs, pos_label=modelo.classes_[1])
        roc_auc = auc(fpr, tpr)
        print(f"AUC-ROC: {roc_auc:.4f}")
        
        # Geração e exportação do gráfico ROC
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Curva ROC - {nome_conjunto}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(f'curva_roc_{nome_conjunto.lower()}.pdf')
        plt.close()

def main(n_gen=3000, pop_size=10): 
    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("mean", np.mean)
    stats.register("max", np.max)

    # Execução do algoritmo genético guardando o logbook
    pop, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.8, mutpb=0.05, ngen=n_gen,
                                       stats=stats, halloffame=hof, verbose=True)

    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    fit_avgs = logbook.select("mean")
    fit_maxs = logbook.select("max")

    historico_df = pd.DataFrame({
        'Geracao': gen,
        'Fitness_Minima': fit_mins,
        'Fitness_Media': fit_avgs,
        'Fitness_Maxima': fit_maxs
    })
    historico_df.to_csv('historico_fitness.csv', index=False)
    print("\nArquivo 'historico_fitness.csv' salvo com sucesso!")

    plt.figure(figsize=(10, 6))
    plt.plot(gen, fit_avgs, label='Média', color='blue', linewidth=2)
    plt.plot(gen, fit_maxs, label='Máxima', color='green', linewidth=2)
    plt.xlabel('Gerações')
    plt.ylabel('Fitness (40% Treino + 60% Validação)')
    plt.title('Evolução da Fitness ao longo das Gerações')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.5)
    plt.savefig('grafico_fitness.pdf')
    plt.close()
    print("Arquivo 'grafico_fitness.pdf' salvo com sucesso!")

    best_ind = hof[0]
    idx = 0

    hidden1 = int(decode_binary(best_ind[idx:idx+n_bits_hidden_layer1], 1, 32, n_bits_hidden_layer1))
    idx += n_bits_hidden_layer1
    
    hidden2 = int(decode_binary(best_ind[idx:idx+n_bits_hidden_layer2], 0, 32, n_bits_hidden_layer2))
    idx += n_bits_hidden_layer2

    activations = ['identity', 'logistic', 'tanh', 'relu']
    solvers = ['lbfgs', 'sgd', 'adam']

    activation_idx = min(int(decode_binary(best_ind[idx:idx+n_bits_activation], 0, len(activations)-1, n_bits_activation)), len(activations)-1)
    idx += n_bits_activation

    solver_idx = min(int(decode_binary(best_ind[idx:idx+n_bits_solver], 0, len(solvers)-1, n_bits_solver)), len(solvers)-1)
    idx += n_bits_solver

    alpha = decode_binary(best_ind[idx:idx+n_bits_alpha], 0.0001, 0.1, n_bits_alpha)
    idx += n_bits_alpha
    lr_init = decode_binary(best_ind[idx:idx+n_bits_lr_init], 0.0001, 0.1, n_bits_lr_init)

    hidden_layers = (hidden1,) if hidden2 == 0 else (hidden1, hidden2)

    print(f'\nMelhor Fitness Encontrada: {best_ind.fitness.values[0]:.4f}')
    print('Melhores parâmetros:')
    print(f'  hidden_layer_sizes = {hidden_layers}')
    print(f'  activation         = {activations[activation_idx]}')
    print(f'  solver             = {solvers[solver_idx]}')
    print(f'  alpha              = {alpha:.5f}')
    print(f'  learning_rate_init = {lr_init:.5f}')

    # Treinamento do modelo final com os melhores parâmetros
    model_final = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activations[activation_idx],
        solver=solvers[solver_idx],
        alpha=alpha,
        learning_rate_init=lr_init,
        max_iter=2000, 
        random_state=42
    )
    model_final.fit(X_train, y_train)

    calcular_metricas(model_final, X_train, y_train, "TREINO")
    calcular_metricas(model_final, X_test, y_test, "TESTE")

    return best_ind

if __name__ == "__main__":
    best = main()