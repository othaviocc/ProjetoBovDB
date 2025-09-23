import os
import pandas as pd
import plotly.graph_objects as go

# Diretório contendo os arquivos com as novas features
directory = r"E:\progamação\vs code\Bovdb\Traiding_Data\main\docs\indicadores" #caminho para pasta onde se encontra teu inidicador 

# Função para gerar as novas colunas
def generate_new_features(df):
    feature_groups = ["SMA", "EMA", "std_open", "std_close"]
    window_sizes = [3, 5, 7, 9]

    for feature in feature_groups:
        for i, larger_window in enumerate(window_sizes):
            for smaller_window in window_sizes[:i]:
                larger_col = f"{feature}_{larger_window}"
                smaller_col = f"{feature}_{smaller_window}"
                new_col = f"{feature}{larger_window}_{smaller_window}"

                if larger_col in df.columns and smaller_col in df.columns:
                    df[new_col] = df[larger_col] - df[smaller_col]

    return df

# Função para plotar o gráfico de candlesticks com SMA e EMA
def plot_candlestick_with_indicators(filepath, date, ticker_id):
    # Carregar o arquivo CSV
    df = pd.read_csv(filepath)

    # Converter a coluna datetime para o formato datetime do pandas
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Filtrar para o dia e ticker_id desejados
    df_day = df[(df["datetime"].dt.date == pd.to_datetime(date).date()) & (df["id_ticker"] == ticker_id)]

    # Verificar se há dados suficientes
    if df_day.empty:
        print(f"Nenhum dado encontrado para o ticker_id {ticker_id} no dia {date}.")
        return

    # Criar o gráfico de candlesticks
    fig = go.Figure(data=[go.Candlestick(
        x=df_day["datetime"],
        open=df_day["open"],
        high=df_day["high"],
        low=df_day["low"],
        close=df_day["close"],
        name="Candlestick"
    )])

    # Adicionar SMA e EMA ao gráfico em um eixo secundário
    sma_columns = [col for col in df.columns if col.startswith("SMA")]
    ema_columns = [col for col in df.columns if col.startswith("EMA")]

    for sma_col in sma_columns:
        fig.add_trace(go.Scatter(
            x=df_day["datetime"],
            y=df_day[sma_col],
            mode="lines",
            name=sma_col,
            yaxis="y2"
        ))

    for ema_col in ema_columns:
        fig.add_trace(go.Scatter(
            x=df_day["datetime"],
            y=df_day[ema_col],
            mode="lines",
            name=ema_col,
            yaxis="y2"
        ))

    # Configurar layout com eixo secundário
    fig.update_layout(
        title=f"Gráfico de Candlesticks com SMA e EMA ({date}, Ticker {ticker_id})",
        xaxis_title="Data e Hora",
        yaxis_title="Preço",
        template="plotly_dark",
        yaxis2=dict(
            title="Indicadores",
            overlaying="y",
            side="right"
        )
    )

    # Mostrar o gráfico
    fig.show()

# Processar um arquivo, um dia e um ticker_id específico
filename = "indicadores_2024_01.csv"  # Altere para o arquivo desejado
filepath = os.path.join(directory, filename)
date = "2024-01-08"  # Altere para o dia desejado
ticker_id = 58413 # Altere para o id_ticker desejado

plot_candlestick_with_indicators(filepath, date, ticker_id)
