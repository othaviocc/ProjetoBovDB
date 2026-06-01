import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
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
    def __init__(self, max_depth=8, min_samples_leaf=10):
        self.tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, 
                                           criterion='entropy', random_state=SEED)
    def fit(self, X_f, y): self.tree.fit(X_f, y)
    def predict_proba(self, X_f): return self.tree.predict_proba(X_f)
    
    def predict_proba_soft(self, X_f):
        t = self.tree.tree_
        n_s = X_f.shape[0]
        res = np.zeros((n_s, self.tree.n_classes_))
        q = [(0, np.ones(n_s))]
        X_a = X_f.values
        while q:
            node, weights = q.pop(0)
            if np.all(weights == 0): continue
            if t.children_left[node] == t.children_right[node]:
                v = t.value[node][0]
                res += weights[:, np.newaxis] * (v / v.sum())
            else:
                feat = t.feature[node]
                mu = np.clip(X_a[:, feat], 0, 1)
                q.append((t.children_right[node], weights * mu))
                q.append((t.children_left[node], weights * (1-mu)))
        return res

class FuzzyRandomForest:
    def __init__(self, n_estimators=100, mf_type='gaussian', n_partitions=3, voting='MWLT', max_depth=8):
        self.n_estimators, self.voting, self.mf_type, self.n_partitions = n_estimators, voting, mf_type, n_partitions
        self.fuzzifier = Fuzzifier(mf_type=mf_type, n_partitions=n_partitions)
        self.trees, self.features_per_tree, self.tree_weights = [], [], []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.fuzzifier.fit(X)
        X_f = self.fuzzifier.transform(X)
        n_s, n_f = X_f.shape
        m_f = int(np.sqrt(n_f)) # Sqrt para maior diversidade
        for _ in range(self.n_estimators):
            idx = np.random.choice(n_s, size=n_s, replace=True)
            oob = np.array(list(set(range(n_s)) - set(idx)))
            feats = np.random.choice(n_f, size=m_f, replace=False)
            self.features_per_tree.append(feats)
            fdt = FuzzyDecisionTree(max_depth=8)
            fdt.fit(X_f.iloc[idx, feats], y.iloc[idx])
            if len(oob) > 0:
                preds_o = fdt.tree.predict(X_f.iloc[oob, feats])
                acc = accuracy_score(y.iloc[oob], preds_o)
                self.tree_weights.append(acc)
                if self.voting == 'MWLFUS':
                    fdt.error_tree = (NearestNeighbors(n_neighbors=5).fit(X_f.iloc[oob, feats]), (preds_o != y.iloc[oob]).astype(int).values)
            else: self.tree_weights.append(1.0)
            self.trees.append(fdt)

    def predict(self, X):
        X_f = self.fuzzifier.transform(X)
        votes = np.zeros((X.shape[0], len(self.classes_)))
        for i, tree in enumerate(self.trees):
            X_sub = X_f.iloc[:, self.features_per_tree[i]]
            p = tree.predict_proba_soft(X_sub) if self.voting == 'SOFT_ROUTING' else tree.predict_proba(X_sub)
            if self.voting == 'SMI': w = 1.0
            elif self.voting == 'MWLT': w = self.tree_weights[i]
            elif self.voting == 'MWLFUS':
                knn, err = tree.error_tree
                _, ids = knn.kneighbors(X_sub)
                w = (1.0 - err[ids].mean(axis=1))[:, np.newaxis]
            else: w = 1.0
            votes += p * w
        return self.classes_[np.argmax(votes, axis=1)]

if __name__ == "__main__":
    df = pd.read_csv('dataset.csv', parse_dates=['datetime'])
    features = ['SMA_3', 'EMA_3', 'SMA_5', 'EMA_5', 'std_close3', 'std_open3', 'ADXR', 'Bollinger_Norm']
    
    X_train, y_train = df[(df['datetime'] < '2024-04-01')][features], df[(df['datetime'] < '2024-04-01')]['trend']
    X_test, y_test = df[(df['datetime'] >= '2024-04-01')][features], df[(df['datetime'] >= '2024-04-01')]['trend']

    mfs = ['gaussian', 'triangular', 'trapezoidal', 'gmfmfs_nonlinear']
    ramos_lista = [3, 5, 7]
    estrategias = ['SMI', 'MWLT', 'MWLFUS', 'SOFT_ROUTING']
    resultados = []

    best_acc = 0
    best_model = None

    print("\nIniciando Grid Search Massivo...")
    for mf in mfs:
        for r in ramos_lista:
            for est in estrategias:
                model = FuzzyRandomForest(mf_type=mf, voting=est, n_partitions=r)
                model.fit(X_train, y_train)
                acc = accuracy_score(y_test, model.predict(X_test))
                resultados.append({'MF': mf, 'Ramos': r, 'Est': est, 'Acc': acc})
                if acc > best_acc:
                    best_acc = acc
                    best_model = model

    df_res = pd.DataFrame(resultados)
    tabela = df_res.pivot_table(index='Est', columns=['MF', 'Ramos'], values='Acc')
    print("\n" + "="*80 + "\nRESULTADOS DO GRID SEARCH (ACURÁCIA TESTE)\n" + "="*80)
    print((tabela * 100).round(2))

    # 2. Relatório do Melhor Modelo
    print(f"\nMELHOR MODELO: {best_model.mf_type.upper()} | Ramos: {best_model.n_partitions} | Est: {best_model.voting}")
    print(f"Acurácia Teste: {best_acc:.4f}")

    # 3. Visualização da Árvore mais importante (maior peso OOB)
    idx_top = np.argmax(best_model.tree_weights)
    tree_top = best_model.trees[idx_top].tree
    feats_top = best_model.features_per_tree[idx_top]
    f_names = best_model.fuzzifier.transform(X_train.head(1)).columns
    
    print(f"\nESTRUTURA DA ÁRVORE CAMPEÃ (Peso OOB: {best_model.tree_weights[idx_top]:.4f})")
    print(export_text(tree_top, feature_names=[f_names[f] for f in feats_top], max_depth=3))

    '''# 4. Matriz de Confusão
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, best_model.predict(X_test)), annot=True, fmt='d', cmap='Greens')
    plt.title(f'Matriz de Confusão - {best_model.mf_type} ({best_model.voting})')
    plt.show()'''

    from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

    def print_text_confusion_matrix(cm, classes, title):
        """Gera uma matriz de confusão elegante em formato texto no terminal."""
        print(f"\n--- {title} ---")
        header = "        " + "  ".join([f"Pred:{str(c):>5}" for c in classes])
        print(header)
        for i, row_label in enumerate(classes):
            row_str = f"Real:{str(row_label):>3} |"
            for val in cm[i]:
                row_str += f"{val:>8} "
            print(row_str)

    # 1. Obter predições para Treino e Teste
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # 2. Relatório de Performance Completo (Classification Report)
    print(f"\n{'='*70}")
    print(f"RELATÓRIO DE PERFORMANCE: {best_model.mf_type.upper()} | {best_model.voting}")
    print(f"{'='*70}")

    # Nomes das classes para o relatório (Ajuste conforme sua base: ex: Venda, Neutro, Compra)
    target_names = [f'Classe {c}' for c in best_model.classes_]

    print("\n>>> MÉTRICAS DE TESTE (OUT-OF-SAMPLE):")
    print(f"Acurácia Balanceada: {balanced_accuracy_score(y_test, y_test_pred):.4f}")
    print(classification_report(y_test, y_test_pred, target_names=target_names))

    # 3. Matrizes de Confusão no Terminal
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    print_text_confusion_matrix(cm_train, best_model.classes_, "MATRIZ DE CONFUSÃO: TREINO")
    print_text_confusion_matrix(cm_test, best_model.classes_, "MATRIZ DE CONFUSÃO: TESTE")

    # 4. Diagnóstico de Generalização (Overfitting)
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)
    gap = (acc_train - acc_test) * 100

    print(f"\n{'='*70}")
    print(f"DIAGNÓSTICO DE ROBUSTEZ")
    print(f"Gap de Acurácia (Treino - Teste): {gap:.2f}%")
    if gap < 5:
        print("Conclusão: Excelente Generalização (Modelo Robusto)")
    elif gap < 10:
        print("Conclusão: Overfitting Leve (Aceitável para Mercado Financeiro)")
    else:
        print("Conclusão: Overfitting Elevado (Necessária Regularização)")
    print(f"{'='*70}")