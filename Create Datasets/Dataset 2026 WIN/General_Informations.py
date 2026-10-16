import pandas as pd
import sqlite3

# Carregar os dados
df = pd.read_csv('dados.csv')
# Conecta ao arquivo do banco
conn = sqlite3.connect('bovdb-v3.1-internal.1.sqlite')

# Converter data para datetime para manipulação mensal
df['date_dt'] = pd.to_datetime(df['date'])
df['mes'] = df['date_dt'].dt.to_period('M')

# Lista de tickers alvo
tickers_alvo = [58431, 58432, 58433]

# Filtrar a base
df_pesquisa = df[df['id_ticker'].isin(tickers_alvo)].copy()

# Agrupamento por Ticker e Mês
resumo_mensal = df_pesquisa.groupby(['id_ticker', 'mes']).agg(
    data_inicial=('date', 'min'),
    data_final=('date', 'max'),
    qtd_barras_1min=('time', 'count'),
    volume_financeiro=('volume', 'sum'),
    contratos=('amount_stock', 'sum')
).reset_index()

print("RESUMO MENSAL POR TICKER")
print(resumo_mensal.to_string(index=False))

# Contagem do número de meses ativos por ticker
contagem_meses = df_pesquisa.groupby('id_ticker')['mes'].nunique().reset_index()
contagem_meses.columns = ['id_ticker', 'total_meses_ativos']

print("TOTAL DE MESES ATIVOS POR TICKER")
print(contagem_meses.to_string(index=False))

# Consulta a tabela ticker para os 3 IDs
query = "SELECT id_ticker, ticker FROM ticker WHERE id_ticker IN (58431, 58432, 58433);"
df_tickers = pd.read_sql_query(query, conn)

print("IDENTIFICAÇÃO DOS TICKERS")
print(df_tickers.to_string(index=False))

conn.close()