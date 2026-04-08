import pandas as pd
import numpy as np
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                             precision_score, recall_score, confusion_matrix)
from deap import base, creator, tools, algorithms

# Se a biblioteca efficient_kan estiver instalada
from efficient_kan import KAN


print("Carregando e preparando os dados...")
# Substitua 'dataset.csv' pelo caminho real do seu arquivo
df = pd.read_csv('dataset.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

features = ['EMA_3', 'SMA_3', 'ADXR', 'Bollinger_Norm', 'EMA_5', 'SMA_5', 'std_close3', 'std_open3']
target = 'trend'  

train_start, train_end = '2024-01-01', '2024-03-30'
test_start, test_end = '2024-04-01', '2024-06-30'

treino = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()
validacao = df[(df['datetime'] >= test_start) & (df['datetime'] <= test_end)].copy()

X_train_raw, y_train_raw = treino[features].values, treino[target].values
X_test_raw, y_test_raw = validacao[features].values, validacao[target].values

# Redes Neurais (como KAN) exigem dados normalizados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Convertendo para Tensores do PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_raw, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_raw, dtype=torch.long)

# Limites dos hiperparâmetros que o GA vai buscar
# [Camadas Ocultas, Learning Rate, Grid Size (específico do KAN)]
BOUNDS_LOW =  [8,  0.0001, 3]
BOUNDS_HIGH = [64, 0.05,  10]

def decodificar_cromossomo(individuo):
    """Mapeia os genes (valores de 0 a 1) para os limites reais dos hiperparâmetros"""
    hidden_dim = int(individuo[0] * (BOUNDS_HIGH[0] - BOUNDS_LOW[0]) + BOUNDS_LOW[0])
    lr = individuo[1] * (BOUNDS_HIGH[1] - BOUNDS_LOW[1]) + BOUNDS_LOW[1]
    grid_size = int(individuo[2] * (BOUNDS_HIGH[2] - BOUNDS_LOW[2]) + BOUNDS_LOW[2])
    return hidden_dim, lr, grid_size

def avaliar_kan(individuo):
    """Função de Fitness que treina o KAN e retorna a acurácia (ou F1)"""
    hidden_dim, lr, grid_size = decodificar_cromossomo(individuo)
    
    # Parâmetros da rede
    input_dim = len(features)
    output_dim = len(np.unique(y_train_raw)) # Ex: 2 para classificação binária
    
    # Instanciando o modelo KAN
    model = KAN([input_dim, hidden_dim, output_dim], grid_size=grid_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epochs = 20 # Mantido baixo para viabilizar as 3000 gerações do GA
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
    # Validação (Fitness)
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs.data, 1)
        # Retornamos a acurácia. A vírgula é obrigatória pois o DEAP espera uma tupla.
        acc = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
        
    return (acc,)


# Queremos maximizar a métrica de avaliação (peso 1.0)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Gerador de atributos: números float entre 0.0 e 1.0
toolbox.register("attr_float", random.random)
# Inicializador do cromossomo com 3 genes (hidden_dim, lr, grid_size)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operadores Genéticos (Baseados na sua Tabela 1)
toolbox.register("evaluate", avaliar_kan)
toolbox.register("mate", tools.cxTwoPoint)                         # Crossover: TwoPoint
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2) # Mutação
toolbox.register("select", tools.selTournament, tournsize=3)       # Seleção: Tournament


def rodar_otimizacao():
    print("\nIniciando Algoritmo Genético...")
    start_time = time.time()
    
    # Parâmetros da Tabela 1
    pop = toolbox.population(n=20)
    CXPB, MUTPB, NGEN = 0.8, 0.05, 3000
    
    # Rastreamento de estatísticas (opcional, para visualização)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    # eaSimple executa o ciclo evolutivo padrão
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, 
                                       ngen=NGEN, stats=stats, verbose=True)
    
    end_time = time.time()
    tempo_total = end_time - start_time
    
    melhor_individuo = tools.selBest(pop, k=1)[0]
    melhor_fitness = melhor_individuo.fitness.values[0]
    
    print(f"\nOtimização concluída em {tempo_total:.2f} segundos.")
    return melhor_individuo, tempo_total

melhor_cromossomo, tempo_otimizacao = rodar_otimizacao()


# Extraindo os melhores hiperparâmetros encontrados
best_hidden, best_lr, best_grid = decodificar_cromossomo(melhor_cromossomo)
print("\n=== Melhores Hiperparâmetros Encontrados (KAN) ===")
print(f"Camadas Ocultas (Hidden Dim): {best_hidden}")
print(f"Taxa de Aprendizado (LR): {best_lr:.5f}")
print(f"Tamanho do Grid (Grid Size): {best_grid}")

# Treinando o modelo final com mais épocas usando os melhores hiperparâmetros
print("\nTreinando o modelo final para extração de métricas...")
final_model = KAN([len(features), best_hidden, len(np.unique(y_train_raw))], grid_size=best_grid)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(final_model.parameters(), lr=best_lr)

final_model.train()
for epoch in range(100): # Aqui usamos mais épocas para o treino real
    optimizer.zero_grad()
    outputs = final_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# Avaliação no conjunto de Teste
final_model.eval()
with torch.no_grad():
    test_outputs = final_model(X_test_tensor)
    # Probabilidades para o ROC AUC
    probs = torch.softmax(test_outputs, dim=1).numpy()
    # Classes preditas
    _, predicted = torch.max(test_outputs.data, 1)
    
    y_test_numpy = y_test_tensor.numpy()
    y_pred_numpy = predicted.numpy()

# Cálculo das Métricas de Mercado e ML
acc = accuracy_score(y_test_numpy, y_pred_numpy)
prec = precision_score(y_test_numpy, y_pred_numpy, average='weighted')
rec = recall_score(y_test_numpy, y_pred_numpy, average='weighted')
f1 = f1_score(y_test_numpy, y_pred_numpy, average='weighted')
conf_matrix = confusion_matrix(y_test_numpy, y_pred_numpy)

# ROC AUC exige formatações diferentes dependendo se é binário ou multiclasse
try:
    if len(np.unique(y_train_raw)) == 2:
        roc_auc = roc_auc_score(y_test_numpy, probs[:, 1])
    else:
        roc_auc = roc_auc_score(y_test_numpy, probs, multi_class='ovr')
except ValueError:
    roc_auc = "N/A (Apenas uma classe presente no teste)"


print("\n" + "="*50)
print("RELATÓRIO DE DESEMPENHO - GA + KAN (5 MINUTOS)")
print("="*50)
print(f"Tempo de Otimização (GA): {tempo_otimizacao:.2f} segundos")
print(f"Acurácia:     {acc:.4f}")
print(f"Precisão:     {prec:.4f}")
print(f"Recall:       {rec:.4f}")
print(f"F1-Score:     {f1:.4f}")
print(f"ROC AUC:      {roc_auc if isinstance(roc_auc, str) else f'{roc_auc:.4f}'}")
print("-" * 50)
print("Matriz de Confusão:")
print(conf_matrix)
print("="*50)


print("\nExtraindo previsões e features no tempo para o KAN...")

# Criando o DataFrame com as features normalizadas que a rede KAN enxergou no teste
df_temporal_kan = pd.DataFrame(X_test, columns=features)

# Adicionando a linha do tempo e as previsões (lembrando que no script do KAN, 
# a base de teste final estava na variável 'validacao')
df_temporal_kan['datetime'] = validacao['datetime'].values
df_temporal_kan['target_real'] = y_test_numpy
df_temporal_kan['previsao'] = y_pred_numpy

# Adicionando colunas de probabilidade se for classificação binária
if len(np.unique(y_train_raw)) == 2:
    df_temporal_kan['prob_classe_0'] = probs[:, 0]
    df_temporal_kan['prob_classe_1'] = probs[:, 1]

# Reorganizando as colunas
colunas_iniciais = ['datetime', 'target_real', 'previsao']
if len(np.unique(y_train_raw)) == 2:
    colunas_iniciais += ['prob_classe_0', 'prob_classe_1']

colunas_finais_kan = colunas_iniciais + features
df_temporal_kan = df_temporal_kan[colunas_finais_kan]

# Salvando o CSV
nome_arquivo_kan = 'kan_previsoes_features_5min.csv'
df_temporal_kan.to_csv(nome_arquivo_kan, index=False)
print(f"Arquivo '{nome_arquivo_kan}' salvo com sucesso!")