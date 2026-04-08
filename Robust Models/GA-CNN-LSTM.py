import numpy as np
import pandas as pd
import random
import warnings

# --- VOLTAMOS PARA O JEITO PADRÃO (Agora que o TF está instalado) ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
# --------------------------------------------------------------------

from deap import base, creator, tools, algorithms
from sklearn.metrics import accuracy_score, confusion_matrix

# Configurações de log
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==========================================
# 1. PREPARAÇÃO DOS DADOS (Sliding Window)
# ==========================================
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    # Cria sequências: Para cada ponto, olha 'time_steps' para trás
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# Carrega os dados
df = pd.read_csv('normalizados_passo2.csv', parse_dates=['datetime'])

# Features selecionadas (baseado no seu código)
features = ['EMA_3', 'SMA_3', 'ADXR', 'Bollinger_Norm', 'EMA_5', 'SMA_5', 
            'std_close3', 'std_open3', 'std_close5', 'std_open5', 'std_open7', 
            'SMA_7', 'std_close7', 'SMA_9', 'std_open9', 'std_close11', 
            'EMA_11', 'std_close9']
target = 'trend'

# Configuração da Janela de Tempo (IMPORTANTE)
# 12 candles de 5 min = 1 hora de visão para trás
TIME_STEPS = 12 

# Separação Temporal (Treino vs Teste)
train_start, train_end = '2024-01-01', '2024-03-30'
test_start, test_end = '2024-04-01', '2024-06-30'

# Filtra os dados
df_train = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()
df_test = df[(df['datetime'] >= test_start) & (df['datetime'] <= test_end)].copy()

# Cria as janelas deslizantes (Formato 3D para CNN-LSTM)
# O X agora terá o formato: (Amostras, 12, N_Features)
X_train, y_train = create_dataset(df_train[features], df_train[target], TIME_STEPS)
X_test, y_test = create_dataset(df_test[features], df_test[target], TIME_STEPS)

print(f"Formato X_train: {X_train.shape} (Amostras, Passos, Features)")
print(f"Formato y_train: {y_train.shape}")

# ==========================================
# 2. CONFIGURAÇÃO DO GA (DEAP)
# ==========================================

# Mapeamento dos Genes (Binário -> Parâmetro Real)
# Definimos quantos bits cada gene vai usar
gene_conf = {
    'n_filters':   {'bits': 3, 'vals': [16, 32, 64, 128, 256, 32, 64, 128]}, # CNN Filtros
    'kernel_size': {'bits': 2, 'vals': [2, 3, 5, 7]},                      # Tamanho do filtro
    'lstm_units':  {'bits': 3, 'vals': [20, 50, 75, 100, 128, 150, 50, 100]}, # Neurônios LSTM
    'dense_units': {'bits': 2, 'vals': [16, 32, 64, 128]},                    # Camada Densa Intermediária
    'dropout':     {'bits': 2, 'vals': [0.1, 0.2, 0.3, 0.5]},                 # Taxa de Dropout
    'batch_size':  {'bits': 2, 'vals': [16, 32, 64, 128]}                     # Batch Size
}

CHROMOSOME_SIZE = sum(g['bits'] for g in gene_conf.values())

creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximizar Acurácia
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=CHROMOSOME_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def decode_individual(individual):
    """Converte o array binário (genes) em parâmetros reais"""
    params = {}
    current_idx = 0
    
    for key, conf in gene_conf.items():
        bits = conf['bits']
        # Pega a fatia de bits correspondente
        gene_bits = individual[current_idx : current_idx + bits]
        # Converte binário para inteiro
        gene_int = int("".join(map(str, gene_bits)), 2)
        # Garante que o índice existe na lista de valores (módulo)
        val_idx = gene_int % len(conf['vals'])
        params[key] = conf['vals'][val_idx]
        
        current_idx += bits
        
    return params

def eval_genome(individual):
    """Função de Fitness: Cria a CNN-LSTM e treina"""
    params = decode_individual(individual)
    
    # Monta a Arquitetura Híbrida
    model = Sequential()
    
    # 1. CNN (Extração de Features)
    model.add(Conv1D(filters=params['n_filters'], 
                     kernel_size=params['kernel_size'], 
                     activation='relu', 
                     input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    
    # 2. LSTM (Análise Temporal)
    model.add(LSTM(params['lstm_units'], return_sequences=False))
    model.add(Dropout(params['dropout']))
    
    # 3. Classificação
    model.add(Dense(params['dense_units'], activation='relu'))
    model.add(Dense(1, activation='sigmoid')) # Saída Binária (0 ou 1)
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    # Early Stopping para não perder tempo se o modelo for ruim
    es = EarlyStopping(monitor='val_loss', patience=3, verbose=0, restore_best_weights=True)
    
    # Treina o modelo (Epochs baixo para o GA ser rápido)
    # Usamos parte do treino como validação interna
    history = model.fit(X_train, y_train, 
                        epochs=5,  # Poucas épocas durante a busca do GA
                        batch_size=params['batch_size'], 
                        validation_split=0.2, 
                        callbacks=[es], 
                        verbose=0)
    
    # Avalia na base de validação interna
    val_acc = max(history.history['val_accuracy'])
    
    return (val_acc,)

toolbox.register("evaluate", eval_genome)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# ==========================================
# 3. LOOP PRINCIPAL (GA)
# ==========================================
def main():
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    pop = toolbox.population(n=10) # População pequena para teste rápido
    ngen = 3                       # Poucas gerações para teste
    
    print(f"Iniciando GA com População={len(pop)} e Gerações={ngen}...")
    
    # Roda o Algoritmo Genético
    final_pop = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=ngen, verbose=True)
    
    # Pega o melhor indivíduo
    best_ind = tools.selBest(pop, 1)[0]
    best_params = decode_individual(best_ind)
    
    print("\n=============================================")
    print("MELHOR INDIVÍDUO ENCONTRADO:")
    print(best_params)
    print(f"Fitness (Validação): {best_ind.fitness.values[0]:.4f}")
    print("=============================================\n")
    
    # ==========================================
    # 4. TREINAMENTO FINAL (MODELO OTIMIZADO)
    # ==========================================
    print("Treinando modelo final com os melhores parâmetros...")
    
    model_final = Sequential()
    model_final.add(Conv1D(filters=best_params['n_filters'], 
                           kernel_size=best_params['kernel_size'], 
                           activation='relu', 
                           input_shape=(X_train.shape[1], X_train.shape[2])))
    model_final.add(MaxPooling1D(pool_size=2))
    model_final.add(LSTM(best_params['lstm_units'], return_sequences=False))
    model_final.add(Dropout(best_params['dropout']))
    model_final.add(Dense(best_params['dense_units'], activation='relu'))
    model_final.add(Dense(1, activation='sigmoid'))
    
    model_final.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    # No treino final, usamos mais épocas
    model_final.fit(X_train, y_train, 
                    epochs=20, 
                    batch_size=best_params['batch_size'], 
                    validation_data=(X_test, y_test), 
                    verbose=1)
    
    # Predições finais
    y_pred_prob = model_final.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int) # Converte probabilidade para 0 ou 1
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nAcurácia Final no Teste: {acc:.4f}")
    print("Matriz de Confusão:")
    print(cm)

if __name__ == "__main__":
    main()