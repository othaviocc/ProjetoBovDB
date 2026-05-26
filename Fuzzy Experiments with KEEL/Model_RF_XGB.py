import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Modelos
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Validação Cruzada e Métricas
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score

# Explicabilidade
from sklearn.tree import export_text
import lime
import lime.lime_tabular

df = pd.read_csv('dataset.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

features = ['EMA_3', 'SMA_3', 'ADXR', 'Bollinger_Norm', 'EMA_5', 'SMA_5', 'std_close3', 'std_open3']
target = 'trend'  

for col in features:
    col_min = df[col].min()
    col_max = df[col].max()
    # Se tiver valores negativos ou acima de 1, aplica a normalização Min-Max
    if col_min < 0 or col_max > 1:
        df[col] = (df[col] - col_min) / (col_max - col_min)

train_mask = (df['datetime'] >= '2024-01-01') & (df['datetime'] <= '2024-03-30')
test_mask = (df['datetime'] >= '2024-04-01') & (df['datetime'] <= '2024-06-30')

X_train, y_train = df.loc[train_mask, features], df.loc[train_mask, target]
X_test, y_test = df.loc[test_mask, features], df.loc[test_mask, target]

def evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    print(f"\n{'='*50}\nAvaliação do Modelo: {model_name}\n{'='*50}")
    
    skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)
    
    print("--- Acurácia na Validação Cruzada (9 Folds) ---")
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        # Treinando e validando no fold atual
        X_fold_train, y_fold_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_fold_val, y_fold_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        model.fit(X_fold_train, y_fold_train)
        fold_pred = model.predict(X_fold_val)
        fold_acc = accuracy_score(y_fold_val, fold_pred)
        cv_scores.append(fold_acc)
        print(f"Fold {fold}: {fold_acc:.4f}")
        
    print(f"-> Acurácia Média CV: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})\n")
    
    # Treinamento final com todo o conjunto de treino e predição no teste
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculando Métricas Finais no Teste
    # 'macro' é usado caso o trend tenha mais de 2 classes. Se for binário, funciona igual.
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print("--- Métricas no Conjunto de Teste (Dados Inéditos) ---")
    print(f"Acurácia : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("Matriz de Confusão:\n", cm)
    
    return model

# Instanciando Random Forest e XGBoost
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, eval_metric='logloss', random_state=42)

# Avaliando RF
rf_model = evaluate_model(rf_model, "Random Forest", X_train, y_train, X_test, y_test)

# Avaliando XGBoost 
xgb_model = evaluate_model(xgb_model, "XGBoost", X_train, y_train, X_test, y_test)


print(f"\n{'='*50}\nEXPLICABILIDADE: RANDOM FOREST\n{'='*50}")
# 4.1 Feature Importance (VARimp)
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
print("--- Importância das Variáveis (Global) ---")
print(importances)

# 4.2 Extração de Regras 
print("\n--- Regras Lógicas de uma das Árvores (IF/THEN) ---")
# Pegamos a primeira árvore (índice 0) para mostrar as regras
tree_rules = export_text(rf_model.estimators_[0], feature_names=features)
print(tree_rules)


print(f"\n{'='*50}\nEXPLICABILIDADE: XGBOOST (com LIME)\n{'='*50}")
# 4.3 Feature Importance Nativo do XGBoost
print("Plotando a importância das variáveis do XGBoost...")
xgb.plot_importance(xgb_model, importance_type='weight', title='Importância (XGBoost)')
plt.show()

# 4.4 Explicabilidade Local com LIME
print("\n--- Explicando uma previsão específica com LIME ---")
# Criando o explicador tabular
# class_names devem corresponder aos rótulos únicos do seu 'trend'
classes = np.unique(y_train).astype(str).tolist() 
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=features,
    class_names=classes,
    mode='classification'
)

# Escolhemos a 1 amostra do conjunto de teste para explicar
instancia_idx = 0
amostra = X_test.iloc[instancia_idx]

# Gerando a explicação (LIME altera um pouco os dados ao redor da amostra para ver como o modelo reage)
exp = explainer.explain_instance(
    data_row=amostra.values,
    predict_fn=xgb_model.predict_proba,
    num_features=5 # Mostra o top 5 variáveis que mais influenciaram a decisão
)

print(f"Explicando a amostra de Teste no índice {instancia_idx}:\nValores reais da amostra:")
print(amostra.to_dict())

print("\nRegras criadas pelo LIME para esta decisão:")
# O LIME gera regras baseadas em limiares, similar ao Fuzzy (ex: se EMA_3 > X e SMA_5 <= Y)
for rule in exp.as_list():
    print(f"Regra LIME: {rule[0]}  --> Peso/Influência na decisão: {rule[1]:.4f}")

# Opcional: Salvar a explicação LIME em um arquivo HTML para visualizar graficamente
exp.save_to_file('lime_explanation.html')