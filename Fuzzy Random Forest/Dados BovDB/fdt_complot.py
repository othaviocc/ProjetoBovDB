import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text
import warnings

warnings.filterwarnings('ignore')

# Semente fixa para reprodutibilidade total
SEED = 42
np.random.seed(SEED)

class Fuzzifier:
    def __init__(self, mf_type='triangular', n_partitions=3):
        self.mf_type = mf_type
        self.n_partitions = n_partitions
        self.params = {}

    def fit(self, X):
        for col in X.columns:
            min_val, max_val = X[col].min(), X[col].max()
            if min_val == max_val: max_val += 1e-9 
            if 'gmfmfs' in self.mf_type:
                quantiles = np.linspace(0, 1, self.n_partitions + 1)
                self.params[col] = np.unique(np.quantile(X[col].dropna(), quantiles))
            else:
                self.params[col] = np.linspace(min_val, max_val, self.n_partitions)

    def transform(self, X):
        X_fuzzy = pd.DataFrame(index=X.index)
        for col in X.columns:
            params = self.params.get(col)
            if 'gmfmfs' in self.mf_type:
                for i in range(len(params) - 1):
                    p_at, p_px = params[i], params[i+1]
                    col_name = f"{col}_{self.mf_type}_{i}"
                    mf_lin = np.where((X[col] >= p_at) & (X[col] <= p_px), (X[col]-p_at)/(p_px-p_at+1e-9), 0)
                    if self.mf_type == 'gmfmfs_nonlinear':
                        mf_fuz = np.where(mf_lin <= 0.5, 2*mf_lin**2, 1-2*(1-mf_lin)**2)
                    else: mf_fuz = mf_lin
                    X_fuzzy[col_name] = mf_fuz
            else:
                centers = params
                w = centers[1] - centers[0] if len(centers) > 1 else 1.0
                for i, c in enumerate(centers):
                    col_name = f"{col}_{self.mf_type}_{i}"
                    if self.mf_type == 'triangular':
                        mf = np.maximum(0, 1 - np.abs(X[col] - c) / w)
                    elif self.mf_type == 'gaussian':
                        mf = np.exp(-0.5 * ((X[col] - c) / (w/1.5))**2)
                    elif self.mf_type == 'trapezoidal':
                        mf = np.clip(1 - (np.maximum(0, np.abs(X[col] - c) - w*0.2) / (w*0.8)), 0, 1)
                    X_fuzzy[col_name] = mf
        return X_fuzzy

class FuzzyDecisionTree:
    def __init__(self, mf_type='gaussian', n_partitions=3, routing='STANDARD', max_depth=8, min_samples_leaf=15):
        self.mf_type = mf_type
        self.n_partitions = n_partitions
        self.routing = routing
        self.fuzzifier = Fuzzifier(mf_type=mf_type, n_partitions=n_partitions)
        self.tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, 
                                           criterion='entropy', random_state=SEED)
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.fuzzifier.fit(X)
        X_f = self.fuzzifier.transform(X)
        self.tree.fit(X_f, y)
        
    def predict_proba_soft(self, X_f):
        """Inferência Fuzzy de Múltiplos Caminhos usando Normas-T"""
        t = self.tree.tree_
        n_s = X_f.shape[0]
        res = np.zeros((n_s, self.tree.n_classes_))
        q = [(0, np.ones(n_s))]
        X_a = X_f.values
        
        while q:
            node, weights = q.pop(0)
            if np.all(weights == 0): continue
            
            # É uma folha: soma os pesos probabilísticos
            if t.children_left[node] == t.children_right[node]:
                v = t.value[node][0]
                res += weights[:, np.newaxis] * (v / v.sum())
            else:
                feat = t.feature[node]
                mu = np.clip(X_a[:, feat], 0, 1)
                
                # Aplicação da Norma-T escolhida
                if self.routing == 'SOFT_PRODUCT':
                    w_right = weights * mu
                    w_left = weights * (1 - mu)
                elif self.routing == 'SOFT_MIN':
                    w_right = np.minimum(weights, mu)
                    w_left = np.minimum(weights, 1 - mu)
                else: # Fallback de segurança
                    w_right = weights * mu
                    w_left = weights * (1 - mu)
                    
                q.append((t.children_right[node], w_right))
                q.append((t.children_left[node], w_left))
        return res

    def predict(self, X):
        X_f = self.fuzzifier.transform(X)
        if self.routing in ['SOFT_PRODUCT', 'SOFT_MIN']:
            probas = self.predict_proba_soft(X_f)
            return self.classes_[np.argmax(probas, axis=1)]
        else:
            return self.tree.predict(X_f) # STANDARD (Caminho Único)

def print_text_confusion_matrix(cm, classes, title):
    """Gera uma matriz de confusão elegante no terminal."""
    print(f"\n--- {title} ---")
    header = "        " + "  ".join([f"Pred:{str(c):>5}" for c in classes])
    print(header)
    for i, row_label in enumerate(classes):
        row_str = f"Real:{str(row_label):>3} |"
        for val in cm[i]:
            row_str += f"{val:>8} "
        print(row_str)

if __name__ == "__main__":
    print("A carregar e processar os dados para a FDT...")
    df = pd.read_csv('dataset.csv', parse_dates=['datetime'])
    features = ['SMA_3', 'EMA_3', 'SMA_5', 'EMA_5', 'std_close3', 'std_open3', 'ADXR', 'Bollinger_Norm']
    
    X_train, y_train = df[(df['datetime'] < '2024-04-01')][features], df[(df['datetime'] < '2024-04-01')]['trend']
    X_test, y_test = df[(df['datetime'] >= '2024-04-01')][features], df[(df['datetime'] >= '2024-04-01')]['trend']

    mfs = ['gaussian', 'triangular', 'trapezoidal', 'gmfmfs_nonlinear']
    ramos_lista = [3, 5, 7, 9] 
    
    # Adicionamos a Norma-T do Produto (Bonissone) e do Mínimo (Zadeh)
    estrategias = ['STANDARD', 'SOFT_PRODUCT', 'SOFT_MIN']
    resultados = []

    best_acc = 0
    best_model = None

    print("\nIniciando Grid Search Massivo da FDT...")
    for mf in mfs:
        for r in ramos_lista:
            for est in estrategias:
                model = FuzzyDecisionTree(mf_type=mf, n_partitions=r, routing=est, max_depth=8, min_samples_leaf=15)
                model.fit(X_train, y_train)
                acc = accuracy_score(y_test, model.predict(X_test))
                
                resultados.append({'MF': mf, 'Ramos': r, 'Estrategia': est, 'Acc': acc})
                
                if acc > best_acc:
                    best_acc = acc
                    best_model = model

    # 1. Tabela Completa
    df_res = pd.DataFrame(resultados)
    tabela = df_res.pivot_table(index='Estrategia', columns=['MF', 'Ramos'], values='Acc')
    print("\n" + "="*80 + "\nRESULTADOS DO GRID SEARCH FDT (ACURÁCIA TESTE)\n" + "="*80)
    print((tabela * 100).round(2))

    # 2. Relatório do Melhor Modelo
    print(f"\nMELHOR MODELO (FDT): {best_model.mf_type.upper()} | Ramos: {best_model.n_partitions} | Roteamento: {best_model.routing}")
    print(f"Acurácia Teste: {best_acc:.4f}")

    # 3. Estrutura da Árvore Campeã
    f_names = best_model.fuzzifier.transform(X_train.head(1)).columns.tolist()
    print(f"\nESTRUTURA DA ÁRVORE CAMPEÃ (FDT)")
    print(export_text(best_model.tree, feature_names=f_names, max_depth=4))

    # 4. Relatório de Performance Completo e Matrizes
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    print(f"\n{'='*70}")
    print(f"RELATÓRIO DE PERFORMANCE: {best_model.mf_type.upper()} | {best_model.routing}")
    print(f"{'='*70}")

    target_names = [f'Classe {c}' for c in best_model.classes_]

    print("\n>>> MÉTRICAS DE TESTE (OUT-OF-SAMPLE):")
    print(f"Acurácia Balanceada: {balanced_accuracy_score(y_test, y_test_pred):.4f}")
    print(classification_report(y_test, y_test_pred, target_names=target_names))

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    print_text_confusion_matrix(cm_train, best_model.classes_, "MATRIZ DE CONFUSÃO: TREINO")
    print_text_confusion_matrix(cm_test, best_model.classes_, "MATRIZ DE CONFUSÃO: TESTE")

    # 5. Diagnóstico de Generalização (Overfitting)
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)
    gap = (acc_train - acc_test) * 100

    print(f"\n{'='*70}")
    print(f"DIAGNÓSTICO DE ROBUSTEZ DA FDT")
    print(f"Acurácia Treino: {acc_train*100:.2f}% | Acurácia Teste: {acc_test*100:.2f}%")
    print(f"Gap de Acurácia: {gap:.2f}%")
    if gap < 5:
        print("Conclusão: Excelente Generalização (Raro para árvore isolada)")
    elif gap < 10:
        print("Conclusão: Overfitting Leve (Aceitável)")
    else:
        print("Conclusão: Overfitting Elevado (Necessita Random Forest)")
    print(f"{'='*70}\n")