import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                             precision_score, recall_score, confusion_matrix)

# Bibliotecas para Otimização e Rede
from mealpy.swarm_based.SMA import OriginalSMA
from efficient_kan import KAN

print("Carregando e preparando os dados...")
df = pd.read_csv('dataset.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

features = ['EMA_3', 'SMA_3', 'ADXR', 'Bollinger_Norm', 'EMA_5', 'SMA_5', 'std_close3', 'std_open3']
target = 'trend'  

# Divisão temporal rigorosa (Evitando Overfitting de Otimização)
train_start, train_end = '2024-01-01', '2024-02-28'
val_start, val_end = '2024-03-01', '2024-03-31'
test_start, test_end = '2024-04-01', '2024-06-30'

treino = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()
validacao = df[(df['datetime'] >= val_start) & (df['datetime'] <= val_end)].copy()
teste = df[(df['datetime'] >= test_start) & (df['datetime'] <= test_end)].copy()

X_train_raw, y_train_raw = treino[features].values, treino[target].values
X_val_raw, y_val_raw = validacao[features].values, validacao[target].values
X_test_raw, y_test_raw = teste[features].values, teste[target].values

# Padronização (Obrigatório para Redes KAN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

# Tensores PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_raw, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_raw, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_raw, dtype=torch.long)

# Limites: [Hidden Dim, Learning Rate, Grid Size]
BOUNDS_LOW =  [8,  0.0001, 3]
BOUNDS_HIGH = [64, 0.05,  10]

def avaliar_kan_sma(solution):
    """Função de Fitness para o SMA: Treina no Treino, testa na Validação"""
    # Decodificando a solução contínua
    hidden_dim = int(solution[0])
    lr = solution[1]
    grid_size = int(solution[2])
    
    input_dim = len(features)
    output_dim = len(np.unique(y_train_raw))
    
    # Instanciando o modelo KAN
    model = KAN([input_dim, hidden_dim, output_dim], grid_size=grid_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epochs = 15 # Épocas reduzidas para o fitness ser mais rápido
    
    # Treino
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
        val_outputs = model(X_val_tensor)
        _, predicted = torch.max(val_outputs.data, 1)
        acc = accuracy_score(y_val_tensor.numpy(), predicted.numpy())
        
    return acc # O SMA buscará maximizar esta acurácia

print("\nIniciando otimização Slime Mould Algorithm (SMA) com KAN...")
start_time = time.time()

# Configuração do problema para o Mealpy
problem_dict = {
    "obj_func": avaliar_kan_sma,
    "bounds": {"lower": BOUNDS_LOW, "upper": BOUNDS_HIGH}, 
    "minmax": "max", 
    "log_to": None   
}

# Configurações fiéis à sua Tabela 1 de parâmetros
EPOCHS = 3000 
POP_SIZE = 20 

# Única linha que muda na inicialização em relação ao GWO
modelo_sma = OriginalSMA(epoch=EPOCHS, pop_size=POP_SIZE)
best_position, best_fitness = modelo_sma.solve(problem_dict)

tempo_otimizacao = time.time() - start_time

best_hidden = int(best_position[0])
best_lr = best_position[1]
best_grid = int(best_position[2])

print("\n=== Melhores Hiperparâmetros Encontrados (SMA) ===")
print(f"Camadas Ocultas (Hidden Dim): {best_hidden}")
print(f"Taxa de Aprendizado (LR): {best_lr:.5f}")
print(f"Tamanho do Grid (Grid Size): {best_grid}")

print("\nTreinando o modelo definitivo para o Teste Out-of-Sample...")
# Juntando treino e validação para maximizar o histórico do modelo final
X_train_full = torch.tensor(np.vstack((X_train, X_val)), dtype=torch.float32)
y_train_full = torch.tensor(np.concatenate((y_train_raw, y_val_raw)), dtype=torch.long)

final_model = KAN([len(features), best_hidden, len(np.unique(y_train_full))], grid_size=best_grid)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(final_model.parameters(), lr=best_lr)

final_model.train()
for epoch in range(100): # Treinamento longo final
    optimizer.zero_grad()
    outputs = final_model(X_train_full)
    loss = criterion(outputs, y_train_full)
    loss.backward()
    optimizer.step()

final_model.eval()
with torch.no_grad():
    test_outputs = final_model(X_test_tensor)
    probs = torch.softmax(test_outputs, dim=1).numpy()
    _, predicted = torch.max(test_outputs.data, 1)
    
    y_test_numpy = y_test_tensor.numpy()
    y_pred_numpy = predicted.numpy()

acc = accuracy_score(y_test_numpy, y_pred_numpy)
prec = precision_score(y_test_numpy, y_pred_numpy, average='weighted')
rec = recall_score(y_test_numpy, y_pred_numpy, average='weighted')
f1 = f1_score(y_test_numpy, y_pred_numpy, average='weighted')
conf_matrix = confusion_matrix(y_test_numpy, y_pred_numpy)

try:
    if len(np.unique(y_train_full)) == 2:
        roc_auc = roc_auc_score(y_test_numpy, probs[:, 1])
    else:
        roc_auc = roc_auc_score(y_test_numpy, probs, multi_class='ovr')
except ValueError:
    roc_auc = "N/A"

print("\n" + "="*50)
print("RELATÓRIO FINAL OOS - SMA + KAN")
print("="*50)
print(f"Tempo de Otimização (SMA): {tempo_otimizacao:.2f} segundos")
print(f"Acurácia no Teste: {acc:.4f}")
print(f"Precisão:          {prec:.4f}")
print(f"Recall:            {rec:.4f}")
print(f"F1-Score:          {f1:.4f}")
print(f"ROC AUC:           {roc_auc if isinstance(roc_auc, str) else f'{roc_auc:.4f}'}")
print("-" * 50)
print("Matriz de Confusão:")
print(conf_matrix)
print("="*50)

print("\nExtraindo previsões e features no tempo para o KAN...")

# Criando o DataFrame para análise temporal
df_temporal_kan = pd.DataFrame(X_test, columns=features)
df_temporal_kan['datetime'] = teste['datetime'].values
df_temporal_kan['target_real'] = y_test_numpy
df_temporal_kan['previsao'] = y_pred_numpy

# Adicionando probabilidades se for um problema binário
if len(np.unique(y_train_raw)) == 2:
    df_temporal_kan['prob_classe_0'] = probs[:, 0]
    df_temporal_kan['prob_classe_1'] = probs[:, 1]

# Ordenando as colunas
colunas_iniciais = ['datetime', 'target_real', 'previsao']
if len(np.unique(y_train_raw)) == 2:
    colunas_iniciais += ['prob_classe_0', 'prob_classe_1']

colunas_finais_kan = colunas_iniciais + features
df_temporal_kan = df_temporal_kan[colunas_finais_kan]

nome_arquivo_kan = 'sma_kan_previsoes_features_5min.csv'
df_temporal_kan.to_csv(nome_arquivo_kan, index=False)
print(f"Arquivo temporal '{nome_arquivo_kan}' salvo com sucesso! Pronto para análise de comportamento de mercado.")