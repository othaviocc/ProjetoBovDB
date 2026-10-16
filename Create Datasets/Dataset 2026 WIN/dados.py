import sqlite3
import pandas as pd

# Conecta ao banco de dados interno
db_path = 'bovdb-v3.1-internal.1.sqlite'
conn = sqlite3.connect(db_path)

# Query que filtra 2026, acha o mais volumoso do mês e extrai as colunas de 1 minuto
query = """
WITH VolumeMensal AS (
    -- Calcula o volume total de cada ticker WIN por mês em 2026
    SELECT 
        p.id_ticker,
        strftime('%Y-%m', p.date) AS mes,
        SUM(p.volume) AS volume_total
    FROM price1 p
    JOIN ticker t ON p.id_ticker = t.id_ticker
    WHERE t.ticker LIKE 'WIN%' AND p.date LIKE '2026-%'
    GROUP BY p.id_ticker, mes
),
RankingMensal AS (
    -- Ranqueia os tickers de cada mês pelo volume (1 é o maior)
    SELECT 
        id_ticker,
        mes,
        RANK() OVER(PARTITION BY mes ORDER BY volume_total DESC) as rank
    FROM VolumeMensal
),
TopTickers AS (
    -- Filtra apenas o ticker mais volumoso (Rank 1) de cada mês
    SELECT id_ticker, mes
    FROM RankingMensal
    WHERE rank = 1
)
-- Puxa todas as colunas de 1 minuto apenas para os vencedores em seus respectivos meses
SELECT 
    p.id_ticker,
    p.date,
    p.time,
    p.open,
    p.close,
    p.high,
    p.low,
    p.average,
    p.volume,
    p.business,
    p.amount_stock
FROM price1 p
JOIN TopTickers t 
    ON p.id_ticker = t.id_ticker 
    AND strftime('%Y-%m', p.date) = t.mes
ORDER BY p.date ASC, p.time ASC;
"""

print("Executando a consulta no banco de dados. Isso pode levar alguns segundos...")
df_win = pd.read_sql_query(query, conn)

# Salva o resultado no arquivo CSV solicitado
csv_filename = 'dados.csv'
df_win.to_csv(csv_filename, index=False)

print(f"Sucesso! Foram extraídas {len(df_win)} linhas de 1 minuto.")
print(f"O arquivo '{csv_filename}' foi salvo na mesma pasta com todas as colunas solicitadas.")

conn.close()