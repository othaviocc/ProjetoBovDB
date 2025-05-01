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

    def calculate_bollinger_bands(self, df, window=20, std_dev=2):
        # Calcular Média Móvel Simples (SMA) e Bandas de Bollinger
        df['SMA'] = df['close'].rolling(window=window).mean()
        df['Upper Band'] = df['SMA'] + (df['close'].rolling(window=window).std() * std_dev)
        df['Lower Band'] = df['SMA'] - (df['close'].rolling(window=window).std() * std_dev)
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
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA'],
        line=dict(color='blue', width=1),
        name='SMA'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Upper Band'],
        line=dict(color='red', width=1),
        name='Upper Band'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Lower Band'],
        line=dict(color='green', width=1),
        name='Lower Band'
    ))

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
query = """
    SELECT id_ticker, date, time, open, close, high, low, average, volume, business, amount_stock
    FROM price5
    WHERE id_ticker = 107 AND date = '2024-06-27'
    """
# Criar uma instância do DataProcessor
processor = DataProcessor(db_path, query)
processor.load_data()
df = processor.process_data()

# Identificar candles de 5 minutos e calcular Bandas de Bollinger
df_5min = processor.identify_5_min_candles()
df_5min = processor.calculate_bollinger_bands(df_5min)

# Plotar o gráfico com Bandas de Bollinger
plot_candlestick_with_bollinger(df_5min)
