import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import warnings

warnings.filterwarnings('ignore')

class Fuzzifier:
    def __init__(self, mf_type='triangular', n_partitions=3):
        self.mf_type = mf_type
        self.n_partitions = n_partitions
        self.params = {}

    def fit(self, X):
        for col in X.columns:
            min_val, max_val = X[col].min(), X[col].max()
            if min_val == max_val: max_val += 1e-9 # Previne divisão por zero
            
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
    def __init__(self, max_depth=8, min_samples_leaf=15):
        self.tree = DecisionTreeClassifier(
            max_depth=max_depth, 
            min_samples_leaf=min_samples_leaf, 
            criterion='entropy',
            random_state=42 # Fixamos a semente para garantir resultados sempre iguais
        )

    def fit(self, X_fuzzy, y):
        self.tree.fit(X_fuzzy, y)
        
    def predict(self, X_fuzzy):
        return self.tree.predict(X_fuzzy)


if __name__ == "__main__":
    
    print("A carregar e processar os dados para a FDT...")
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

    
    FUNCOES_PARA_TESTAR = ['triangular', 'trapezoidal', 'gaussian', 'gmfmfs_linear', 'gmfmfs_nonlinear']
    RAMOS_PARA_TESTAR = [3, 5, 7, 9] # Adicionei 9 ramos aqui pois a FDT isolada é mais rápida de testar
    PROFUNDIDADE_MAXIMA = 8 # Ajuste este valor se a árvore estiver a dar overfitting

    print("\n" + "="*70)
    print("A INICIAR O GRID SEARCH DA ÁRVORE DE DECISÃO FUZZY (FDT)")
    print("="*70)

    resultados_fdt = []

    for mf in FUNCOES_PARA_TESTAR:
        print(f"\n[{mf.upper()}] A treinar modelos...")
        
        for ramos in RAMOS_PARA_TESTAR:
            
            # Fuzzifica os dados com a combinação atual
            fuzz = Fuzzifier(mf_type=mf, n_partitions=ramos)
            X_train_fdt = fuzz.fit_transform(X_train)
            X_valid_fdt = fuzz.transform(X_valid)

            # Treina a FDT
            fdt = FuzzyDecisionTree(max_depth=PROFUNDIDADE_MAXIMA)
            fdt.fit(X_train_fdt, y_train)
            
            # Avalia fora da amostra (Out-of-sample)
            acc = accuracy_score(y_valid, fdt.predict(X_valid_fdt))
            print(f"  -> {ramos} Ramos: Acurácia = {acc:.4f}")
            
            resultados_fdt.append({
                'Funcao_Pertinencia': mf,
                'Ramos': ramos,
                'Acuracia': acc
            })

    print("\n" + "="*70)
    print("RESUMO COMPLETO DOS RESULTADOS (FDT ISOLADA)")
    print("="*70)
    
    df_resultados = pd.DataFrame(resultados_fdt)
    
    # Cria uma tabela pivotada pronta para o artigo
    tabela_pivot = df_resultados.pivot_table(
        index=['Funcao_Pertinencia'], 
        columns=['Ramos'], 
        values='Acuracia'
    )
    
    tabela_formatada = (tabela_pivot * 100).round(2)    
    tabela_formatada.columns.name = 'Nº de Ramos'
    
    print(tabela_formatada)
    print("="*70)
    
    nome_ficheiro = "resultados_fdt_isolada.csv"
    tabela_formatada.to_csv(nome_ficheiro)
    print(f"\nResultados guardados com sucesso no ficheiro: '{nome_ficheiro}'")