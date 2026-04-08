import pandas as pd
import numpy as np
import time
import random
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                             precision_score, recall_score, confusion_matrix)
from deap import base, creator, tools, algorithms
from pytorch_tabnet.tab_model import TabNetClassifier


print("Carregando e preparando os dados...")
df = pd.read_csv('dataset.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

features = ['EMA_3', 'SMA_3', 'ADXR', 'Bollinger_Norm', 'EMA_5', 'SMA_5', 'std_close3', 'std_open3']
target = 'trend'  

# DIVISÃO EM 3 PARTES PARA EVITAR DATA LEAKAGE
train_start, train_end = '2024-01-01', '2024-02-28' # Usado para treinar os pesos
val_start, val_end = '2024-03-01', '2024-03-31'     # Usado pelo GA para o fitness
test_start, test_end = '2024-04-01', '2024-06-30'   # O verdadeiro Out-of-Sample (intocável)

treino = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()
validacao = df[(df['datetime'] >= val_start) & (df['datetime'] <= val_end)].copy()
teste = df[(df['datetime'] >= test_start) & (df['datetime'] <= test_end)].copy()

# O TabNet espera arrays numpy, não tensores nativos do PyTorch
X_train, y_train = treino[features].values, treino[target].values
X_val, y_val = validacao[features].values, validacao[target].values
X_test, y_test = teste[features].values, teste[target].values

# Redes profundas necessitam de dados padronizados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# Limites dos hiperparâmetros (TabNet)
# [n_d (largura), n_steps (passos de decisão), Learning Rate]
BOUNDS_LOW =  [8,  3, 0.001]
BOUNDS_HIGH = [64, 10, 0.05]

def decodificar_cromossomo_tabnet(individuo):
    """Mapeia os genes (0 a 1) para os limites reais do TabNet"""
    # No TabNet, n_d e n_a geralmente têm o mesmo valor para estabilidade
    n_d = int(individuo[0] * (BOUNDS_HIGH[0] - BOUNDS_LOW[0]) + BOUNDS_LOW[0])
    n_steps = int(individuo[1] * (BOUNDS_HIGH[1] - BOUNDS_LOW[1]) + BOUNDS_LOW[1])
    lr = individuo[2] * (BOUNDS_HIGH[2] - BOUNDS_LOW[2]) + BOUNDS_LOW[2]
    return n_d, n_steps, lr

def avaliar_tabnet(individuo):
    """Função de Fitness: Treina na base Treino, avalia na base Validação"""
    n_d, n_steps, lr = decodificar_cromossomo_tabnet(individuo)
    
    # Instanciando o TabNet
    clf = TabNetClassifier(
        n_d=n_d, n_a=n_d, n_steps=n_steps,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=lr),
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=0 # Silencia o print de treinamento a cada geração
    )
    
    # Treinando o modelo (com um máximo de epochs baixo para a otimização)
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=['validacao'],
        eval_metric=['accuracy'],
        max_epochs=15, 
        patience=5,
        batch_size=1024, # Ajuste conforme a memória da sua GPU/RAM
        virtual_batch_size=128
    )
    
    # Calculando a acurácia na base de VALIDAÇÃO (O modelo nunca vê o teste aqui)
    preds_val = clf.predict(X_val)
    acc = accuracy_score(y_val, preds_val)
        
    return (acc,)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Parâmetros da sua Tabela 1
toolbox.register("evaluate", avaliar_tabnet)
toolbox.register("mate", tools.cxTwoPoint)                         
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2) 
toolbox.register("select", tools.selTournament, tournsize=3)       

def rodar_otimizacao():
    print("\nIniciando Algoritmo Genético com TabNet...")
    start_time = time.time()
    
    pop = toolbox.population(n=20)
    CXPB, MUTPB, NGEN = 0.8, 0.05, 3000
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, 
                                       ngen=NGEN, stats=stats, verbose=True)
    
    end_time = time.time()
    tempo_total = end_time - start_time
    
    melhor_individuo = tools.selBest(pop, k=1)[0]
    return melhor_individuo, tempo_total

melhor_cromossomo, tempo_otimizacao = rodar_otimizacao()


best_n_d, best_n_steps, best_lr = decodificar_cromossomo_tabnet(melhor_cromossomo)
print("\n=== Melhores Hiperparâmetros Encontrados (TabNet) ===")
print(f"Largura (n_d = n_a): {best_n_d}")
print(f"Passos (n_steps): {best_n_steps}")
print(f"Taxa de Aprendizado (LR): {best_lr:.5f}")

print("\nTreinando o modelo definitivo para extração de métricas de mercado...")
final_model = TabNetClassifier(
    n_d=best_n_d, n_a=best_n_d, n_steps=best_n_steps,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=best_lr),
    verbose=0
)

# Agora podemos juntar Treino + Validação para ter mais histórico antes de prever o Teste
X_train_full = np.vstack((X_train, X_val))
y_train_full = np.concatenate((y_train, y_val))

final_model.fit(
    X_train=X_train_full, y_train=y_train_full,
    max_epochs=100, # Treino longo e definitivo
    batch_size=1024,
    virtual_batch_size=128
)

# A GRANDE HORA DA VERDADE: Prevendo nos dados de teste intocados!
preds_test = final_model.predict(X_test)
probs_test = final_model.predict_proba(X_test)

acc = accuracy_score(y_test, preds_test)
prec = precision_score(y_test, preds_test, average='weighted')
rec = recall_score(y_test, preds_test, average='weighted')
f1 = f1_score(y_test, preds_test, average='weighted')
conf_matrix = confusion_matrix(y_test, preds_test)

try:
    if len(np.unique(y_train_full)) == 2:
        roc_auc = roc_auc_score(y_test, probs_test[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, probs_test, multi_class='ovr')
except ValueError:
    roc_auc = "N/A"

print("\n" + "="*50)
print("RELATÓRIO FINAL OOS (Out-Of-Sample) - GA + TABNET")
print("="*50)
print(f"Tempo de Otimização (GA): {tempo_otimizacao:.2f} segundos")
print(f"Acurácia no Teste: {acc:.4f}")
print(f"Precisão:          {prec:.4f}")
print(f"Recall:            {rec:.4f}")
print(f"F1-Score:          {f1:.4f}")
print(f"ROC AUC:           {roc_auc if isinstance(roc_auc, str) else f'{roc_auc:.4f}'}")
print("-" * 50)
print("Matriz de Confusão:")
print(conf_matrix)
print("="*50)


print("\nExtraindo a importância temporal das features no TabNet...")

# O método explain() retorna a matriz de importância de cada feature para CADA linha
matriz_importancia, dicionario_mascaras = final_model.explain(X_test)

# Criando o DataFrame com os pesos das features
df_temporal_tabnet = pd.DataFrame(matriz_importancia, columns=features)

# Adicionando a linha do tempo e os resultados
df_temporal_tabnet['datetime'] = teste['datetime'].values
df_temporal_tabnet['target_real'] = y_test
df_temporal_tabnet['previsao'] = preds_test

# Reorganizando para o datetime ficar na frente, facilitando o plot
colunas_finais = ['datetime', 'target_real', 'previsao'] + features
df_temporal_tabnet = df_temporal_tabnet[colunas_finais]

# Salvando o CSV
nome_arquivo_tabnet = 'tabnet_importancias_5min.csv'
df_temporal_tabnet.to_csv(nome_arquivo_tabnet, index=False)
print(f"Arquivo '{nome_arquivo_tabnet}' salvo com sucesso!")