import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
import warnings

# Ignora avisos não críticos para manter o terminal limpo durante o Grid Search
warnings.filterwarnings('ignore')

class Fuzzifier:
    def __init__(self, mf_type='triangular', n_partitions=3):
        """
        mf_type suportados: 
        'triangular', 'trapezoidal', 'gaussian', 'gmfmfs_linear', 'gmfmfs_nonlinear'
        """
        self.mf_type = mf_type
        self.n_partitions = n_partitions
        self.params = {}

    def fit(self, X):
        for col in X.columns:
            min_val, max_val = X[col].min(), X[col].max()
            if min_val == max_val: max_val += 1e-9 # Previne divisão por zero
            
            if 'gmfmfs' in self.mf_type:
                # GMFMFS: usa quantis (densidade) para criar intervalos adaptativos
                quantiles = np.linspace(0, 1, self.n_partitions + 1)
                self.params[col] = np.unique(np.quantile(X[col].dropna(), quantiles))
            else:
                # Tradicionais: usam centros e intervalos estáticos
                self.params[col] = np.linspace(min_val, max_val, self.n_partitions)

    def transform(self, X):
        X_fuzzy = pd.DataFrame(index=X.index)
        
        for col in X.columns:
            params = self.params.get(col)
            
            if 'gmfmfs' in self.mf_type:
                for i in range(len(params) - 1):
                    p_atual, p_prox = params[i], params[i+1]
                    col_name = f"{col}_{self.mf_type}_{i}"
                    
                    mf_linear = np.where(
                        (X[col] >= p_atual) & (X[col] <= p_prox),
                        (X[col] - p_atual) / (p_prox - p_atual + 1e-9),
                        0
                    )
                    
                    if self.mf_type == 'gmfmfs_nonlinear':
                        mf_fuzzy = np.where(mf_linear <= 0.5, 2 * mf_linear**2, 1 - 2 * (1 - mf_linear)**2)
                    else:
                        mf_fuzzy = mf_linear
                        
                    X_fuzzy[col_name] = mf_fuzzy
            else:
                centers = params
                w = centers[1] - centers[0] if len(centers) > 1 else 1.0
                
                for i, c in enumerate(centers):
                    col_name = f"{col}_{self.mf_type}_{i}"
                    
                    if self.mf_type == 'triangular':
                        mf = np.maximum(0, 1 - np.abs(X[col] - c) / w)
                        
                    elif self.mf_type == 'gaussian':
                        sigma = w / 1.5 
                        mf = np.exp(-0.5 * ((X[col] - c) / sigma)**2)
                        
                    elif self.mf_type == 'trapezoidal':
                        flat_w = w * 0.2
                        slope_w = w * 0.8
                        mf = np.clip(1 - (np.maximum(0, np.abs(X[col] - c) - flat_w) / slope_w), 0, 1)
                        
                    X_fuzzy[col_name] = mf
                    
        return X_fuzzy

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class FuzzyDecisionTree:
    def __init__(self, max_depth=8, min_samples_leaf=10):
        self.tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, criterion='entropy')
        self.oob_score_ = 0.0
        self.error_tree = None

    def fit(self, X_fuzzy, y):
        self.tree.fit(X_fuzzy, y)
        
    def predict_proba(self, X_fuzzy):
        return self.tree.predict_proba(X_fuzzy)
        
    def predict(self, X_fuzzy):
        return self.tree.predict(X_fuzzy)


class FuzzyRandomForest:
    def __init__(self, n_estimators=100, mf_type='trapezoidal', n_partitions=3, 
                 voting_strategy='MWLFUS', max_depth=8, max_features=None):
        self.n_estimators = n_estimators
        self.voting_strategy = voting_strategy
        self.fuzzifier = Fuzzifier(mf_type=mf_type, n_partitions=n_partitions)
        self.trees = []
        self.features_per_tree = []
        self.tree_weights = []
        self.classes_ = None
        self.max_depth = max_depth
        self.max_features = max_features

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X_fuzzy = self.fuzzifier.fit_transform(X)
        n_samples, n_features = X_fuzzy.shape
        
        # Define o max_features: usa o fornecido ou o padrão (raiz quadrada)
        m_features = self.max_features if self.max_features is not None else int(np.sqrt(n_features))
        
        for i in range(self.n_estimators):
            # Bagging
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            oob_indices = np.array(list(set(range(n_samples)) - set(indices)))
            
            # Random Subspace
            features = np.random.choice(n_features, size=m_features, replace=False)
            self.features_per_tree.append(features)
            
            X_train_boot = X_fuzzy.iloc[indices, features]
            y_train_boot = y.iloc[indices]
            
            fdt = FuzzyDecisionTree(max_depth=self.max_depth)
            fdt.fit(X_train_boot, y_train_boot)
            
            if len(oob_indices) > 0:
                X_oob = X_fuzzy.iloc[oob_indices, features]
                y_oob = y.iloc[oob_indices]
                preds_oob = fdt.predict(X_oob)
                oob_acc = accuracy_score(y_oob, preds_oob)
                fdt.oob_score_ = oob_acc
                self.tree_weights.append(oob_acc)
                
                # CORREÇÃO: Transformação para matriz Numpy (uso do .values)
                if self.voting_strategy == 'MWLFUS':
                    errors = (preds_oob != y_oob).astype(int).values 
                    error_tree = NearestNeighbors(n_neighbors=5)
                    error_tree.fit(X_oob)
                    fdt.error_tree = (error_tree, errors)
            else:
                self.tree_weights.append(1.0)
                
            self.trees.append(fdt)

    def predict(self, X):
        X_fuzzy = self.fuzzifier.transform(X)
        final_votes = np.zeros((X.shape[0], len(self.classes_)))
        
        for t_idx, tree in enumerate(self.trees):
            features = self.features_per_tree[t_idx]
            X_f_subset = X_fuzzy.iloc[:, features]
            probas = tree.predict_proba(X_f_subset)
            
            if self.voting_strategy == 'SMI':
                final_votes += probas * 1.0
                
            elif self.voting_strategy == 'MWLT':
                final_votes += probas * self.tree_weights[t_idx]
                
            elif self.voting_strategy == 'MWLFUS':
                knn, oob_errors = tree.error_tree
                _, indices = knn.kneighbors(X_f_subset)
                
                # CORREÇÃO: Indexação direta na matriz Numpy (sem .iloc)
                local_weights = 1.0 - oob_errors[indices].mean(axis=1)
                weighted_probas = probas * local_weights[:, np.newaxis]
                final_votes += weighted_probas

        best_class_idx = np.argmax(final_votes, axis=1)
        return self.classes_[best_class_idx]


if __name__ == "__main__":
    
    print("A carregar e processar os dados...")
    df = pd.read_csv('dataset.csv', parse_dates=['datetime'])

    # As 8 features de alta performance identificadas
    features_permitidas = [
        'SMA_3', 'EMA_3', 'SMA_5', 'EMA_5', 
        'std_close3', 'std_open3', 'ADXR', 'Bollinger_Norm'
    ]

    # Splits Temporais (Walk-Forward Validation)
    train_start, train_end = '2024-01-01', '2024-03-30'
    test_start, test_end = '2024-04-01', '2024-06-30'

    treino = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()
    validacao = df[(df['datetime'] >= test_start) & (df['datetime'] <= test_end)].copy()

    colunas_para_remover = ['datetime','date','close','open','low','high','volume','average','amount_stock','id_ticker','business']
    for base in [treino, validacao]:
        base.drop(columns=[col for col in colunas_para_remover if col in base.columns], inplace=True, errors='ignore')

    X_train = treino[treino.columns.intersection(features_permitidas)]
    y_train = treino['trend']
    X_valid = validacao[validacao.columns.intersection(features_permitidas)]
    y_valid = validacao['trend']

    print(f"Treino: {X_train.shape[0]} amostras | Validação: {X_valid.shape[0]} amostras")
    print(f"Features ativas ({X_train.shape[1]}):", list(X_train.columns))

    
    FUNCOES_PARA_TESTAR = ['triangular', 'trapezoidal', 'gaussian', 'gmfmfs_nonlinear']
    RAMOS_PARA_TESTAR = [3, 5, 7]
    ESTRATEGIAS_VOTACAO = ['SMI', 'MWLT', 'MWLFUS']
    
    # Ajuste para compensar a baixa dimensionalidade (8 variáveis)
    MAX_FEATURES_AJUSTADO = int(X_train.shape[1] * 0.75) 

    print("\n" + "="*80)
    print("A INICIAR O GRID SEARCH AUTOMATIZADO PARA O ARTIGO")
    print("="*80)

    resultados_gerais = []

    for mf in FUNCOES_PARA_TESTAR:
        print(f"\n[{mf.upper()}] A avaliar Função de Pertinência...")
        
        for ramos in RAMOS_PARA_TESTAR:
            print(f"  -> Granularidade: {ramos} ramos")
            
            # 1. Avalia o Baseline (FDT) para esta combinação
            fuzz_baseline = Fuzzifier(mf_type=mf, n_partitions=ramos)
            X_train_fdt = fuzz_baseline.fit_transform(X_train)
            X_valid_fdt = fuzz_baseline.transform(X_valid)

            fdt = FuzzyDecisionTree(max_depth=8)
            fdt.fit(X_train_fdt, y_train)
            acc_baseline = accuracy_score(y_valid, fdt.predict(X_valid_fdt))
            
            resultados_gerais.append({
                'Funcao_Pertinencia': mf,
                'Ramos': ramos,
                'Modelo/Estrategia': 'FDT Baseline',
                'Acuracia': acc_baseline
            })

            # 2. Avalia o Ensemble (FRF) com as 3 Estratégias
            for estrategia in ESTRATEGIAS_VOTACAO:
                frf = FuzzyRandomForest(
                    n_estimators=100, 
                    mf_type=mf, 
                    n_partitions=ramos, 
                    voting_strategy=estrategia,
                    max_depth=8,
                    max_features=MAX_FEATURES_AJUSTADO
                )
                
                frf.fit(X_train, y_train)
                y_pred_frf = frf.predict(X_valid)
                acc_frf = accuracy_score(y_valid, y_pred_frf)
                
                resultados_gerais.append({
                    'Funcao_Pertinencia': mf,
                    'Ramos': ramos,
                    'Modelo/Estrategia': f'FRF - {estrategia}',
                    'Acuracia': acc_frf
                })

    print("\n" + "="*80)
    print("RESUMO COMPLETO DOS RESULTADOS (GRID SEARCH)")
    print("="*80)
    
    df_resultados = pd.DataFrame(resultados_gerais)
    
    # Cria uma tabela pivotada pronta para o artigo
    tabela_pivot = df_resultados.pivot_table(
        index=['Modelo/Estrategia'], 
        columns=['Funcao_Pertinencia', 'Ramos'], 
        values='Acuracia'
    )
    
    # Multiplica por 100 e formata para % (ex: 60.31%)
    tabela_formatada = (tabela_pivot * 100).round(2)
    
    print(tabela_formatada)
    print("="*80)
    
    # Guarda o resultado num ficheiro CSV
    nome_ficheiro = "resultados_experimento_fuzzy.csv"
    tabela_formatada.to_csv(nome_ficheiro)
    print(f"\nResultados guardados com sucesso no ficheiro: '{nome_ficheiro}'")
    print("Pode importar este ficheiro diretamente para o Excel ou LaTeX!")