import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
import math

np.random.seed(42)

# Ignorar os warnings do KMeans
warnings.filterwarnings('ignore')

# Funções Necessarias
def trapezoidal_membership(x, a, b, c, d):
    """Função de pertinência Trapezoidal"""
    if x <= a or x >= d:
        return 0.0
    elif b <= x <= c:
        return 1.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif c < x < d:
        return (d - x) / (d - c)
    
def fuzzy_entropy(class_weights):
    """Calcula a Entropia Fuzzy ponderada"""
    total = sum(class_weights.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for w in class_weights.values():
        if w > 0:
            p = w / total
            entropy -= p * np.log2(p)
    return entropy

# Estrutura Fuzzy
class FuzzyNode:
    def __init__(self, is_leaf=False):
        self.is_leaf = is_leaf
        self.feature = None
        self.membership_funcs = {}   # baixo, medio, alto (ou mais partições)
        self.children = {}
        self.class_distribution = {}

class FuzzyDecisionTree:
    def __init__(self, 
                 max_depth=5, 
                 min_entropy=0.01, 
                 max_features="sqrt", 
                 min_samples_split=2, 
                 min_samples_leaf=1,
                 min_membership_weight=0.05,
                 n_partitions=3):
        
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.max_features = max_features           # Pode ser 'sqrt', 'log2', um inteiro, ou None (usa todas)
        self.min_samples_split = min_samples_split # Mínimo de dados ponderados para dividir o nó
        self.min_samples_leaf = min_samples_leaf   # Mínimo de dados ponderados para criar uma folha
        self.min_membership_weight = min_membership_weight # Poda Fuzzy: Pertinência mínima para propagar o dado
        self.n_partitions = n_partitions           # Quantos clusters fuzzy por atributo (ex: 3 = Baixo/Médio/Alto)
        self.root = None

    def fit(self, X, y, weights):
        self.n_classes = len(np.unique(y))
        
        # Resolve o max_features
        n_features_total = X.shape[1]
        if self.max_features == "sqrt":
            self.m_feat = max(1, int(np.sqrt(n_features_total)))
        elif self.max_features == "log2":
            self.m_feat = max(1, int(np.log2(n_features_total)))
        elif isinstance(self.max_features, int):
            self.m_feat = min(self.max_features, n_features_total)
        else:
            self.m_feat = n_features_total # Usa todas se None
            
        self.root = self._build_tree(X, y, weights, depth=0)

    def _get_membership_params(self, X, feature_idx):
        """Cria as partições Fuzzy usando KMeans, respeitando o n_partitions"""
        values = X[:, feature_idx].reshape(-1, 1)
        K = min(self.n_partitions, len(np.unique(values))) # Garante que não tenta criar mais clusters que valores únicos
        
        if K < 2:
            return None
            
        kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
        kmeans.fit(values)
        centers = np.sort(kmeans.cluster_centers_.flatten())

        params = {}
        if K == 3:
            c1, c2, c3 = centers
            params = {
                "baixo":  (-np.inf, -np.inf, c1, c2),
                "medio":  (c1, c2, c2, c3),  
                "alto":   (c2, c3, np.inf, np.inf)
            }
        elif K == 2:
            c1, c2 = centers
            params = {
                "baixo": (-np.inf, -np.inf, c1, c2),
                "alto":  (c1, c2, np.inf, np.inf)
            }
        elif K > 3:
            # Lógica generalizada para mais partições (Cria os Trapézios dinamicamente)
            params[f"p_0"] = (-np.inf, -np.inf, centers[0], centers[1])
            for i in range(1, K - 1):
                params[f"p_{i}"] = (centers[i-1], centers[i], centers[i], centers[i+1])
            params[f"p_{K-1}"] = (centers[K-2], centers[K-1], np.inf, np.inf)

        return params

    def _build_tree(self, X, y, weights, depth):
        node = FuzzyNode()
        total_weight_node = sum(weights)

        class_weights = defaultdict(float)
        for label, w in zip(y, weights):
            class_weights[label] += w

        current_entropy = fuzzy_entropy(class_weights)

        # Aplicação dos Critérios de Parada -dos Hiperparâmetros
        if (depth >= self.max_depth or 
            current_entropy < self.min_entropy or 
            len(np.unique(y[weights > 0])) <= 1 or
            total_weight_node < self.min_samples_split): # Regra do min_samples_split aplicada
            
            node.is_leaf = True
            total = sum(class_weights.values())
            node.class_distribution = {c: w / total for c, w in class_weights.items()}
            return node
        
        # Sorteio de Features
        features = np.random.choice(X.shape[1], self.m_feat, replace=False)

        best_feature = None
        best_params = None
        best_gain = -1
        best_splits = None

        for feature in features:
            params = self._get_membership_params(X, feature)
            if not params:
                continue
                
            expected_entropy = 0
            splits = {}
            valid_split = True
            
            for branch, p in params.items():
                new_weights = np.array([w * trapezoidal_membership(xi, *p) for xi, w in zip(X[:, feature], weights)])
                splits[branch] = new_weights
                branch_weight = np.sum(new_weights)
                
                # regra do min_samples_leaf (Verifica se alguma folha futura ficaria muito "vazia")
                if branch_weight > 0 and branch_weight < self.min_samples_leaf:
                    valid_split = False
                    break 

                if branch_weight >= self.min_samples_leaf:
                    branch_class_weights = defaultdict(float)
                    for label, w in zip(y, new_weights):
                        branch_class_weights[label] += w
                    branch_entropy = fuzzy_entropy(branch_class_weights)
                    expected_entropy += (branch_weight / total_weight_node) * branch_entropy
            
            if not valid_split:
                continue # Se a divisão viola min_samples_leaf, pula para a próxima feature
                    
            gain = current_entropy - expected_entropy
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_params = params
                best_splits = splits

        # Se não houve ganho de informação ou não achou split válido
        if best_feature is None or best_gain <= 0.0001:
            node.is_leaf = True
            total = sum(class_weights.values())
            node.class_distribution = {c: w / total for c, w in class_weights.items()}
            return node

        node.feature = best_feature
        node.membership_funcs = best_params

        for branch, new_weights in best_splits.items():
            if new_weights.sum() >= self.min_samples_leaf:
                node.children[branch] = self._build_tree(X, y, new_weights, depth + 1)
            else:
                child = FuzzyNode(is_leaf=True)
                child.class_distribution = {c: w / sum(class_weights.values()) for c, w in class_weights.items()}
                node.children[branch] = child

        return node
    
    def predict_proba(self, x):
        return self._predict_node(self.root, x, weight=1.0)

    def _predict_node(self, node, x, weight):
        if node.is_leaf:
            return {c: weight * p for c, p in node.class_distribution.items()}

        results = defaultdict(float)
        value = x[node.feature]

        for branch, params in node.membership_funcs.items():
            mu = trapezoidal_membership(value, *params)
            
            # poda Fuzzy de Predição (min_membership_weight)
            # Só propaga se a pertinência for significativa, poupando processamento
            if mu >= self.min_membership_weight and branch in node.children:
                child_res = self._predict_node(node.children[branch], x, weight * mu)
                for c, v in child_res.items():
                    results[c] += v

        return results

# Ensenble Fuzzy Random Forest
class FuzzyRandomForest:
    def __init__(self, 
                 n_estimators=100, 
                 max_depth=5, 
                 max_features="sqrt", 
                 min_samples_split=2, 
                 min_samples_leaf=1,
                 max_samples=1.0, 
                 min_membership_weight=0.05,
                 n_partitions=3):
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_samples = max_samples # Fração dos dados usada no Bagging (ex: 0.8)
        self.min_membership_weight = min_membership_weight
        self.n_partitions = n_partitions
        self.trees = []

    def fit(self, X, y):
        n = len(X)
        self.trees = []
        
        # Calcula quantos dados puxar no Bagging
        n_samples_bootstrap = int(n * self.max_samples)
        
        for _ in range(self.n_estimators):
            # Bootstrapping com controle de max_samples
            indices = np.random.choice(n, n_samples_bootstrap, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            weights = np.ones(len(X_boot))

            tree = FuzzyDecisionTree(
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_membership_weight=self.min_membership_weight,
                n_partitions=self.n_partitions
            )
            tree.fit(X_boot, y_boot, weights)
            self.trees.append(tree)

    def predict(self, X):
        predictions = []

        for x in X:
            total = defaultdict(float)

            for tree in self.trees:
                probs = tree.predict_proba(x)
                for c, v in probs.items():
                    total[c] += v

            predictions.append(max(total, key=total.get))

        return np.array(predictions)


# Treino e Validação

if __name__ == "__main__":
    iris = load_iris()  #carrega dataset
    X = iris.data
    y = iris.target

    skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)
    
    train_accuracies = []
    test_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Instanciando o modelo 
        model = FuzzyRandomForest(
            n_estimators=15, 
            max_depth=3,
            max_features="sqrt",
            min_samples_split=4,    # O nó precisa de pelo menos peso 4 para tentar dividir
            min_samples_leaf=1,     # Ramos muito fracos (< 1 de peso) não são criados
            max_samples=0.9,        # Usa 90% dos dados da bag para dar mais aleatoriedade
            min_membership_weight=0.01, # Só segue caminho fuzzy se pertinência > 1%
            n_partitions=3          # Cria os 3 conjuntos (Baixo, Medio, Alto)
        )
        
        # Treinamento
        model.fit(X_train, y_train)
        
        # Prever no Treino
        preds_train = model.predict(X_train)
        acc_train = accuracy_score(y_train, preds_train)
        train_accuracies.append(acc_train)
        
        # Prever no Teste
        preds_test = model.predict(X_test)
        acc_test = accuracy_score(y_test, preds_test)
        test_accuracies.append(acc_test)
        #prints acuracias treinno e  test
        print(f"Fold {fold+1} | Acc Treino: {acc_train:.4f} | Acc Teste: {acc_test:.4f}")

    # Resultado Final
    print(f"Acurácia Média TREINO: {np.mean(train_accuracies):.4f}")
    print(f"Acurácia Média TESTE:  {np.mean(test_accuracies):.4f}")