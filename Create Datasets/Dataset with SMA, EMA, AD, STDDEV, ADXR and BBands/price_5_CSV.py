import sqlite3
import pandas as pd

# Caminho para o banco de dados
db_path = r'C:\\Users\\othav\\BovDB.v2\\Database_define.db'

# Configuração dos meses e IDs
config = {
    "2024_01": {"start_date": "2024-01-01", "end_date": "2024-01-31", "id_tickers": (58413, 2952)},
    "2024_02": {"start_date": "2024-02-01", "end_date": "2024-02-28", "id_tickers": (58413, 2952)},
    "2024_03": {"start_date": "2024-03-01", "end_date": "2024-03-31", "id_tickers": (2952, 2963)},
    "2024_04": {"start_date": "2024-04-01", "end_date": "2024-04-30", "id_tickers": (2952, 2963)},
    "2024_05": {"start_date": "2024-05-01", "end_date": "2024-05-31", "id_tickers": (2963, 2978)},
    "2024_06": {"start_date": "2024-06-01", "end_date": "2024-06-30", "id_tickers": (2963, 2978)},
}

# Conexão com o banco de dados
conn = sqlite3.connect(db_path)

dataframes = []  # Lista para armazenar os dataframes

for month, params in config.items():
    start_date = params["start_date"]
    end_date = params["end_date"]
    id_tickers = params["id_tickers"]

    # Query para filtrar os dados
    query = f"""
    SELECT *
    FROM Price5
    WHERE date BETWEEN ? AND ?
      AND id_ticker IN ({','.join(['?'] * len(id_tickers))})
    """
    
    # Executa a consulta
    df = pd.read_sql_query(query, conn, params=(start_date, end_date, *id_tickers))
    df["month"] = month  # Adiciona uma coluna indicando o mês
    dataframes.append(df)  # Adiciona o dataframe à lista

# Fecha a conexão com o banco de dados
conn.close()

# Concatena todos os dataframes e ordena por data
final_df = pd.concat(dataframes).sort_values(by="date")

final_df = final_df.drop(columns=["month"])

# Caminho do arquivo de saída
output_path = r"dados.csv"

# Salva o dataframe em um único arquivo CSV
final_df.to_csv(output_path, index=False)
print(f"Arquivo único gerado: {output_path}")
print("Processo concluído.")
