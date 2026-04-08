import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Preparação dos Dados
df = pd.read_csv('dataset.csv', parse_dates=['datetime']).sort_values('datetime')

features = [
    'EMA_3', 'SMA_3', 'ADXR', 'Bollinger_Norm', 'EMA_5', 'SMA_5', 
    'std_close3', 'std_open3', 'std_close5', 'std_open5', 'std_open7', 
    'SMA_7', 'std_close7', 'SMA_9', 'std_open9', 'std_close11', 'EMA_11', 'std_close9'
]
target = 'trend'

# Divisão conforme sua lógica
treino_df = df[(df['datetime'] >= '2024-01-01') & (df['datetime'] <= '2024-03-30')].copy()
teste_df = df[(df['datetime'] >= '2024-04-01') & (df['datetime'] <= '2024-06-30')].copy()

X_train, y_train = treino_df[features], treino_df[target]
X_test, y_test = teste_df[features], teste_df[target]

# 2. Configuração DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("n_estimators", random.randint, 100, 1200)
toolbox.register("max_depth", random.randint, 3, 10)
toolbox.register("learning_rate", random.uniform, 0.005, 0.1)
toolbox.register("subsample", random.uniform, 0.4, 0.9)
toolbox.register("gamma", random.uniform, 0, 10.0)
toolbox.register("min_child_weight", random.randint, 1, 40)

toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.n_estimators, toolbox.max_depth, toolbox.learning_rate, 
                  toolbox.subsample, toolbox.gamma, toolbox.min_child_weight), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 3. Função Fitness (Evoluindo sobre o Teste)
def evaluate(individual):
    n_est = max(50, int(individual[0]))
    m_depth = max(2, int(individual[1]))
    l_rate = max(0.001, min(0.5, individual[2]))
    sub = max(0.1, min(1.0, individual[3]))
    gam = max(0, individual[4])
    min_child = max(0, int(individual[5]))
    
    try:
        model = XGBClassifier(
            n_estimators=n_est, max_depth=m_depth, learning_rate=l_rate,
            subsample=sub, gamma=gam, min_child_weight=min_child,
            random_state=42, eval_metric='logloss', n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return accuracy_score(y_test, preds),
    except:
        return 0.0,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# 4. Execução
def run_evolution():
    pop = toolbox.population(n=20)
    ngen = 300
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    
    print("Iniciando Evolução Genética...")
    algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.05, ngen=ngen, 
                        stats=stats, halloffame=hof, verbose=True)
    return hof[0]

if __name__ == "__main__":
    best = run_evolution()
    
    # 5. Extração Final de Métricas (Treino e Teste)
    n_est, m_depth, l_rate, sub, gam, min_child = best
    final_model = XGBClassifier(
        n_estimators=max(50, int(n_est)),
        max_depth=max(2, int(m_depth)),
        learning_rate=max(0.001, l_rate),
        subsample=max(0.1, min(1.0, sub)),
        gamma=max(0, gam),
        min_child_weight=max(0, int(min_child)),
        random_state=42
    )
    final_model.fit(X_train, y_train)
    
    # Predições para ambos
    y_train_pred = final_model.predict(X_train)
    y_test_pred = final_model.predict(X_test)
    
    print("\n" + "="*50)
    print(f"MELHORES PARÂMETROS: {best}")
    print("COM 18 FEATURES")
    
    print(f"\nACURÁCIA DE TREINO: {accuracy_score(y_train, y_train_pred):.4f}")
    print("Matriz de Confusão (Treino):")
    print(confusion_matrix(y_train, y_train_pred))
    
    print(f"\nACURÁCIA DE TESTE: {accuracy_score(y_test, y_test_pred):.4f}")
    print("Matriz de Confusão (Teste):")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nRelatório Final (Teste):")
    print(classification_report(y_test, y_test_pred))