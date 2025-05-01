import sqlite3
import pandas as pd
import plotly.graph_objects as go
import random
from deap import base, creator, tools, algorithms
import numpy as np

class DataProcessor:
    def __init__(self, db_path, query):
        self.db_path = db_path
        self.query = query
        self.df = None

    def load_data(self):
        # Conectar ao banco de dados e carregar os dados
        conn = sqlite3.connect(self.db_path)
        self.df = pd.read_sql_query(self.query, conn)
        conn.close()

    def process_data(self):
        # Criar uma nova coluna de datetime combinando data e hora
        self.df['datetime'] = pd.to_datetime(self.df['date'] + ' ' + self.df['time'], format='%Y-%m-%d %H:%M:%S')
        # Configurar datetime como índice
        self.df.set_index('datetime', inplace=True)
        # Filtrar os dados para incluir apenas a partir das 09:00:00
        self.df = self.df[self.df.index.time >= pd.to_datetime('09:00:00').time()]
        return self.df

    def identify_5_min_candles(self):
        # Reamostragem para manter os candles de 5 minutos
        df_5min = self.df.resample('5T').agg({
            'open': 'first',
            'close': 'last',
            'high': 'max',
            'low': 'min',
            'volume': 'sum'
        }).dropna()
        return df_5min
    
    def identify_60_min_candles(self):
        # Reamostragem para manter os candles de 60 minutos
        df_60min = self.df.resample('60T').agg({
            'open': 'first',
            'close': 'last',
            'high': 'max',
            'low': 'min',
            'volume': 'sum'
        }).dropna()
        return df_60min

    def detectar_topos_fundos_60_min(self):
        # Acessando as funções de candle de 5 e 60 minutos
        df_60min = self.identify_60_min_candles()
        df_5min = self.identify_5_min_candles()

        topos = []
        fundos = []
        pontos_confirmacao = []  # Armazenar pontos de confirmação para topos e fundos

        i = 0
        while i < len(df_60min) - 1:
            candle_atual = df_60min.iloc[i]
            candle_atual_alta = candle_atual['close'] > candle_atual['open']  #define um candle de alta
            candle_atual_baixa = candle_atual['close'] <= candle_atual['open'] #define um candle de baixa

            # Detectando sequência de alta
            if candle_atual_alta:
                sequencia_alta = [candle_atual]

                # Recolher candles consecutivos de alta de 60 minutos
                while i + 1 < len(df_60min) and df_60min.iloc[i + 1]['close'] > df_60min.iloc[i + 1]['open']:
                    i += 1
                    sequencia_alta.append(df_60min.iloc[i])

                # Encontrar o candle com o maior fechamento dentro dessa sequência de alta
                if sequencia_alta:
                    candle_topo = max(sequencia_alta, key=lambda x: x['close'])

                    # Buscar os candles de 5 minutos dentro do intervalo do topo
                    intervalo_topo_5min = df_5min[(df_5min.index >= candle_topo.name) & 
                                                  (df_5min.index < candle_topo.name + pd.Timedelta(minutes=60))]
                    if not intervalo_topo_5min.empty:
                        maior_close_5min = intervalo_topo_5min['close'].max()
                        candle_5min_topo = intervalo_topo_5min[intervalo_topo_5min['close'] == maior_close_5min].iloc[0]
                        topos.append((candle_5min_topo.name, candle_5min_topo['close']))
                        pontos_confirmacao.append((candle_5min_topo.name, candle_5min_topo['close']))

                    # Verificar o próximo intervalo de 60 minutos para um fechamento mais alto
                    if i + 1 < len(df_60min):
                        prox_candle = df_60min.iloc[i + 1]
                        prox_intervalo_5min = df_5min[(df_5min.index >= prox_candle.name) & 
                                                       (df_5min.index < prox_candle.name + pd.Timedelta(minutes=60))]
                        if not prox_intervalo_5min.empty:
                            maior_close_prox_5min = prox_intervalo_5min['close'].max()

                            # Atualizar topo se um fechamento mais alto existir no próximo intervalo
                            if maior_close_prox_5min > maior_close_5min:
                                candle_5min_topo_prox = prox_intervalo_5min[prox_intervalo_5min['close'] == maior_close_prox_5min].iloc[0]
                                topos[-1] = (candle_5min_topo_prox.name, candle_5min_topo_prox['close'])
                                pontos_confirmacao[-1] = (candle_5min_topo_prox.name, candle_5min_topo_prox['close'])

            elif candle_atual_baixa:
                # Detectando sequência de baixa
                sequencia_baixa = [candle_atual]
                while i + 1 < len(df_60min) and df_60min.iloc[i + 1]['close'] < df_60min.iloc[i + 1]['open']:
                    i += 1
                    sequencia_baixa.append(df_60min.iloc[i])

                # Encontrar o candle com o menor fechamento dentro dessa sequência de baixa
                if sequencia_baixa:
                    candle_fundo = min(sequencia_baixa, key=lambda x: x['close'])

                    # Buscar os candles de 5 minutos dentro do intervalo do fundo
                    intervalo_fundo_5min = df_5min[(df_5min.index >= candle_fundo.name) & 
                                                  (df_5min.index < candle_fundo.name + pd.Timedelta(minutes=60))]
                    if not intervalo_fundo_5min.empty:
                        menor_close_5min = intervalo_fundo_5min['close'].min()
                        candle_5min_fundo = intervalo_fundo_5min[intervalo_fundo_5min['close'] == menor_close_5min].iloc[0]
                        fundos.append((candle_5min_fundo.name, candle_5min_fundo['close']))
                        pontos_confirmacao.append((candle_5min_fundo.name, candle_5min_fundo['close']))

                    # Verificar o próximo intervalo de 60 minutos para um fechamento mais baixo
                    if i + 1 < len(df_60min):
                        prox_candle = df_60min.iloc[i + 1]
                        prox_intervalo_5min = df_5min[(df_5min.index >= prox_candle.name) & 
                                                       (df_5min.index < prox_candle.name + pd.Timedelta(minutes=60))]
                        if not prox_intervalo_5min.empty:
                            menor_close_prox_5min = prox_intervalo_5min['close'].min()

                            # Atualizar fundo se um fechamento mais baixo existir no próximo intervalo
                            if menor_close_prox_5min < menor_close_5min:
                                candle_5min_fundo_prox = prox_intervalo_5min[prox_intervalo_5min['close'] == menor_close_prox_5min].iloc[0]
                                fundos[-1] = (candle_5min_fundo_prox.name, candle_5min_fundo_prox['close'])
                                pontos_confirmacao[-1] = (candle_5min_fundo_prox.name, candle_5min_fundo_prox['close'])

            # Avançar para o próximo candle de 60 minutos
            i += 1

        return topos, fundos, pontos_confirmacao


# Configuração do AG
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

n_bits_period = 5
n_bits_std = 9

toolbox.register("attr_bin", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bin, n_bits_period + n_bits_std)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def calculate_fitness(df, period, std_fac):
    df['SMA'] = df['close'].rolling(period).mean()
    df['STD'] = df['close'].rolling(period).std()
    df['Upper Band'] = df['SMA'] + (df['STD'] * std_fac)
    df['Lower Band'] = df['SMA'] - (df['STD'] * std_fac)

    topos, fundos, _ = processor.detectar_topos_fundos_60_min()
    
    gain = 0
    penalty = 0
    
    for topo in topos:
        if topo[1] > df.loc[topo[0], 'Upper Band']:
            gain += 1
    for fundo in fundos:
        if fundo[1] < df.loc[fundo[0], 'Lower Band']:
            gain += 1

    for idx, row in df.iterrows():
        if idx not in [topo[0] for topo in topos] and idx not in [fundo[0] for fundo in fundos]:
            if row['close'] > row['Upper Band']:
                penalty += 1
            elif row['close'] < row['Lower Band']:
                penalty += 1

    print("ganho", gain)
    print("penalidade", penalty)

    fitness = (gain) - (penalty * 0.01)

    return fitness

def decode_binary(gene, lower, upper, n_bits):
    binary_str = ''.join(map(str, gene))
    int_value = int(binary_str, 2)
    max_value = 2 ** n_bits - 1
    return lower + (int_value / max_value) * (upper - lower)

def evaluate(individual):
    period = int(decode_binary(individual[:n_bits_period], 2, 30, n_bits_period))
    std_fac = decode_binary(individual[n_bits_std:], 0.6, 4, n_bits_std)
    return calculate_fitness(df_5min.copy(), period, std_fac),

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main(df, n_gen=1, pop_size=20):
    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("mean", np.mean)
    stats.register("max", np.max)

    algorithms.eaSimple(population, 
                        toolbox, 
                        cxpb=0.8, 
                        mutpb=0.05, 
                        ngen=n_gen, 
                        stats=stats, 
                        halloffame=hof, 
                        verbose=True)
    
    best_individual = hof[0]
    best_period = int(decode_binary(best_individual[:n_bits_period], 2, 30, n_bits_period))
    best_std_fac = decode_binary(best_individual[n_bits_std:], 0.6 , 4, n_bits_std)
    print(f"Melhor indivíduo encontrado: Period = {best_period}, STD Factor = {best_std_fac}")

    return best_period, best_std_fac

# Plotar as Bandas de Bollinger ajustadas
def plot_candlestick_with_bollinger(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candles'
    ))
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['Upper Band'], 
        line=dict(color='green', width=1),
        name='Upper Band'
    )) 
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['Lower Band'], 
        line=dict(color='red', width=1),
        name='Lower Band'
    ))
    fig.update_layout(title='Candlestick com Bandas de Bollinger Otimizadas', 
                      xaxis_title='Data', 
                      yaxis_title='Preço',
                      xaxis_rangeslider_visible=False
                      )
    fig.show()

db_path = r'C:\\Users\\othav\\BovDB.v2\\Database_define.db' 
query = """
    SELECT id_ticker, date, time, open, close, high, low, average, volume, business, amount_stock
FROM price5
WHERE 
    (id_ticker = 58413 AND date BETWEEN '2024-01-01' AND '2024-01-31')
    OR 
    (id_ticker = 2952 AND date BETWEEN '2024-02-01' AND '2024-03-31');
"""


processor = DataProcessor(db_path, query)
processor.load_data()
df = processor.process_data()
df_5min = processor.identify_5_min_candles()

best_period, best_std_fac = main(df_5min)
#Melor periodo Bollinger bands
df_5min['SMA'] = df_5min['close'].rolling(best_period).mean()
df_5min['STD'] = df_5min['close'].rolling(best_period).std()
df_5min['Upper Band'] = df_5min['SMA'] + (df_5min['STD'] * best_std_fac)
df_5min['Lower Band'] = df_5min['SMA'] - (df_5min['STD'] * best_std_fac)


df_5min = df_5min[df_5min.index.date == pd.to_datetime('2024-03-16').date()]

plot_candlestick_with_bollinger(df_5min)