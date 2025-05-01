import sqlite3
import pandas as pd
import plotly.graph_objects as go

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

    def calculate_bollinger_bands(self, df, period=7, std_fac=0.7929549):
        df['SMA'] = df['close'].rolling(period).mean()    #Media Movel Simples --> Middle Band
        df['STD'] = df['close'].rolling(period).std()     #Desvio  Padrão
        df['Upper Band'] = df['SMA'] + (df['STD'] * std_fac)   #Upper Band
        df['Lower Band'] = df['SMA'] - (df['STD'] * std_fac)   #Lower Band

        return df

# Função para plotar candles com Bandas de Bollinger
def plot_candlestick_with_bollinger(df):
    fig = go.Figure()

    # Plotar os candles
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candles'
    ))

    # Adicionar SMA, Banda Superior e Banda Inferior
    '''fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA'],
        line=dict(color='blue', width=1),
        name='SMA'
    ))'''
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

    '''min_lower_band = df['Lower Band'].min()
    min_lower_index = df['Lower Band'].idxmin()
    fig.add_trace(go.Scatter(
        x=[min_lower_index],  # Ponto do mínimo da Lower Band
        y=[min_lower_band],
        marker=dict(color='purple', size=10),
        name='Min Lower Band',
    ))

    max_upper_band = df['Upper Band'].max()
    max_upper_index = df['Upper Band'].idxmax()
    fig.add_trace(go.Scatter(
        x=[max_upper_index],  # Ponto do máximo da Upper Band
        y=[max_upper_band],
        marker=dict(color='orange', size=10),
        name='Max Upper Band',
    ))
'''
    # Configurar layout do gráfico
    fig.update_layout(
        title='Candlestick com Bandas de Bollinger',
        xaxis_title='Data',
        yaxis_title='Preço',
        xaxis_rangeslider_visible=False
    )

    fig.show()

# Exemplo de uso
db_path = r'C:\\Users\\othav\\BovDB.v2\\Database_define.db'
query =  """
    SELECT id_ticker, date, time, open, close, high, low, average, volume, business, amount_stock
FROM price5
WHERE  
    (id_ticker = 2952 AND date BETWEEN '2024-02-01' AND '2024-03-31');
"""
# Criar uma instância do DataProcessor
processor = DataProcessor(db_path, query)
processor.load_data()
df = processor.process_data()

# Identificar candles de 5 minutos e calcular Bandas de Bollinger
df_5min = processor.identify_5_min_candles()
df_5min = processor.calculate_bollinger_bands(df_5min)

# Filtrar apenas o segundo dia para o plot
df_5min = df_5min[df_5min.index.date == pd.to_datetime('2024-04-07').date()]

# Plotar o gráfico com Bandas de Bollinger
plot_candlestick_with_bollinger(df_5min)
