import pandas as pd

# Carrega o arquivo original
df = pd.read_csv('dados.csv')
print(f"Total de linhas originais: {len(df)}")

# Encontra e remove todas as linhas onde a hora começa com '9' e '6' (ex: '90:xx:xx' ou  '60:xx:xx')
# Os horários normais da manhã não são afetados pois começam com '0' (ex: '09:00:00')
df_limpo = df[~df['time'].str.startswith(('9','8','7','6'))].copy()

# Salva o resultado no novo arquivo
df_limpo.to_csv('dados_limpos.csv', index=False)

print(f"Foram removidas {len(df) - len(df_limpo)} linhas com horários anômalos.")
print(f"Arquivo salvo com sucesso. Novo total de linhas: {len(df_limpo)}")