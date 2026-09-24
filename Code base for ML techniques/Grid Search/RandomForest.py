import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, matthews_corrcoef
from sklearn.model_selection import ParameterGrid
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

X_train, y_train = treino[features], treino[target]
X_val, y_val     = validacao[features], validacao[target]
X_test, y_test   = teste[features], teste[target]

param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10],
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy']
}

grid = list(ParameterGrid(param_grid))
print(f"Iniciando Grid Search RF com {len(grid)} combinações...\n")

best_fitness = 0.0
best_params = None

for i, params in enumerate(grid):
    model = RandomForestClassifier(
        **params,
        random_state=42,
        n_jobs=-1
    )
    
    try:
        model.fit(X_train, y_train)
        acc_train = accuracy_score(y_train, model.predict(X_train))
        acc_val = accuracy_score(y_val, model.predict(X_val))
        fitness = (0.4 * acc_train) + (0.6 * acc_val)
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_params = params
            
    except Exception as e:
        continue
        
    if (i + 1) % 100 == 0:
        print(f"Progresso: {i + 1}/{len(grid)} concluídos...")

print(f"\nMelhor Fitness (RF): {best_fitness:.4f}")
print("Melhores parâmetros:")
for k, v in best_params.items():
    print(f"  {k} = {v}")

model_final = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
model_final.fit(X_train, y_train)

def calcular_metricas(modelo, X, y, nome_conjunto, modelo_nome):
    preds = modelo.predict(X)
    probs = modelo.predict_proba(X)[:, 1] if len(modelo.classes_) == 2 else None

    print(f"\n{'='*15} Métricas para {nome_conjunto} ({modelo_nome}) {'='*15}")
    print("Matriz de Confusão:")
    print(confusion_matrix(y, preds))
    print("\nRelatório de Classificação:")
    print(classification_report(y, preds))
    print(f"MCC: {matthews_corrcoef(y, preds):.4f}")

    if probs is not None:
        fpr, tpr, _ = roc_curve(y, probs, pos_label=modelo.classes_[1])
        roc_auc = auc(fpr, tpr)
        print(f"AUC-ROC: {roc_auc:.4f}")
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Curva ROC {modelo_nome} - {nome_conjunto}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(f'curva_roc_{nome_conjunto.lower()}_{modelo_nome.lower()}.pdf')
        plt.close()

calcular_metricas(model_final, X_train, y_train, "TREINO", "RF")
calcular_metricas(model_final, X_test, y_test, "TESTE", "RF")