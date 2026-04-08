import pandas as pd
import numpy as np
import time
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                             precision_score, recall_score, confusion_matrix)

# Bibliotecas para Otimização e Rede
from mealpy.swarm_based.AO import OriginalAO
from pytorch_tabnet.tab_model import TabNetClassifier

print("Carregando e preparando os dados...")
df = pd.read_csv('dataset.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

features = ['EMA_3', 'SMA_3', 'ADXR', 'Bollinger_Norm', 'EMA_5', 'SMA_5', 'std_close3', 'std_open3']
target = 'trend'  

# Separação Temporal: Treino para os pesos da rede, Validação para a metaheurística, Teste intocável
train_start, train_end = '2024-01-01', '2024-02-28'
val_start, val_end = '2024-03-01', '2024-03-31'
test_start, test_end = '2024-04-01', '2024-06-30'

treino = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()
validacao = df[(df['datetime'] >= val_start) & (df['datetime'] <= val_end)].copy()
teste = df[(df['datetime'] >= test_start) & (df['datetime'] <= test_end)].copy()

X_train, y_train = treino[features].values, treino[target].values
X_val, y_val = validacao[features].values, validacao[target].values
X_test, y_test = teste[features].values, teste[target].values

# Padronização (Essencial para Deep Learning)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Limites de busca: [n_d (largura das camadas), n_steps (passos de atenção), Learning Rate]
BOUNDS_LOW =  [8,  3, 0.001]
BOUNDS_HIGH = [64, 10, 0.05]

def avaliar_tabnet_ao(solution):
    """Função de Fitness para a Águia (AO): Treina no Treino, avalia na Validação"""
    # Decodificando a solução contínua 
    n_d = int(solution[0])
    n_steps = int(solution[1])
    lr = solution[2]
    
    # n_a (atenção) é fixado com o mesmo valor de n_d (decisão) por boas práticas de estabilidade
    clf = TabNetClassifier(
        n_d=n_d, n_a=n_d, n_steps=n_steps,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=lr),
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=0 # Silenciado para não poluir o terminal durante as 3000 gerações
    )
    
    # Épocas baixas para que a avaliação da metaheurística não demore meses
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=['validacao'],
        eval_metric=['accuracy'],
        max_epochs=15, 
        patience=5,
        batch_size=1024,
        virtual_batch_size=128
    )
    
    # Previsão na VALIDAÇÃO
    preds_val = clf.predict(X_val)
    acc = accuracy_score(y_val, preds_val)
        
    return acc

print("\nIniciando otimização Aquila Optimizer (AO) com TabNet...")
start_time = time.time()

problem_dict = {
    "obj_func": avaliar_tabnet_ao,
    "bounds": {"lower": BOUNDS_LOW, "upper": BOUNDS_HIGH},
    "minmax": "max", 
    "log_to": None   
}

# Parâmetros espelhados da sua Tabela 1 para garantir comparação científica justa
EPOCHS = 3000
POP_SIZE = 20

modelo_ao = OriginalAO(epoch=EPOCHS, pop_size=POP_SIZE)
best_position, best_fitness = modelo_ao.solve(problem_dict)

tempo_otimizacao = time.time() - start_time

best_n_d = int(best_position[0])
best_n_steps = int(best_position[1])
best_lr = best_position[2]

print("\n=== Melhores Hiperparâmetros Encontrados (AO) ===")
print(f"Largura (n_d = n_a): {best_n_d}")
print(f"Passos (n_steps): {best_n_steps}")
print(f"Taxa de Aprendizado (LR): {best_lr:.5f}")

print("\nTreinando o modelo final definitivo...")
final_model = TabNetClassifier(
    n_d=best_n_d, n_a=best_n_d, n_steps=best_n_steps,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=best_lr),
    verbose=0
)

# Unindo os dados para prover a maior bagagem histórica possível ao modelo antes de encarar o teste
X_train_full = np.vstack((X_train, X_val))
y_train_full = np.concatenate((y_train, y_val))

final_model.fit(
    X_train=X_train_full, y_train=y_train_full,
    max_epochs=100,
    batch_size=1024,
    virtual_batch_size=128
)

# A GRANDE HORA DA VERDADE: Prevendo o futuro intocado
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
print("RELATÓRIO FINAL OOS - AO + TABNET")
print("="*50)
print(f"Tempo de Otimização (AO): {tempo_otimizacao:.2f} segundos")
print(f"Acurácia no Teste: {acc:.4f}")
print(f"Precisão:          {prec:.4f}")
print(f"Recall:            {rec:.4f}")
print(f"F1-Score:          {f1:.4f}")
print(f"ROC AUC:           {roc_auc if isinstance(roc_auc, str) else f'{roc_auc:.4f}'}")
print("-" * 50)
print("Matriz de Confusão:")
print(conf_matrix)
print("="*50)

print("\nGerando CSV de explicabilidade temporal do TabNet...")

# O TabNet extrai as máscaras de atenção que mostram o peso dado a cada indicador técnico a cada 5 minutos
matriz_importancia, _ = final_model.explain(X_test)

df_temporal = pd.DataFrame(matriz_importancia, columns=features)
df_temporal['datetime'] = teste['datetime'].values
df_temporal['target_real'] = y_test
df_temporal['previsao'] = preds_test

# Reordenando para deixar os dados de referência no início da tabela
cols = ['datetime', 'target_real', 'previsao'] + features
df_temporal = df_temporal[cols]

nome_arquivo = 'ao_tabnet_importancias_5min.csv'
df_temporal.to_csv(nome_arquivo, index=False)
print(f"Arquivo '{nome_arquivo}' salvo com sucesso! Pronto para compor os gráficos do seu artigo A1.")