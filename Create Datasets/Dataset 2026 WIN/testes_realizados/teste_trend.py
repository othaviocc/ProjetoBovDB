import pandas as pd
import numpy as np

# 1. Carregar o dataset usando o caminho completo
caminho_arquivo = r'C:\Users\othav\Desktop\repositorio BovDB\Create Datasets\Dataset 2026 WIN\dados_indicadores.csv'
df = pd.read_csv(caminho_arquivo)

# 2. Ordenar por data e horário
df = df.sort_values(by=['date', 'datetime']).reset_index(drop=True)

# 3. Lógica da tendência
df['next_close'] = df.groupby('date')['close'].shift(-1)
df['trend'] = np.where(df['close'] > df['next_close'], 0, 1)

# Tratamento para o final do dia
df.loc[df['next_close'].isna(), 'trend'] = np.nan

# Excluir a coluna temporária
df = df.drop(columns=['next_close'])

# 4. SALVAR DE VOLTA NO ARQUIVO CSV
# O index=False evita que o Pandas crie uma coluna extra de numeração
df.to_csv(caminho_arquivo, index=False)

print("Coluna 'trend' calculada e arquivo atualizado com sucesso!")