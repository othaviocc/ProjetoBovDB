import pandas as pd
import numpy as np

def create_keel_dat(csv_path, output_path):
    features = ['EMA_3', 'SMA_3', 'ADXR', 'Bollinger_Norm', 'EMA_5', 'SMA_5', 'std_close3', 'std_open3']
    target = 'trend'
    
    df = pd.read_csv(csv_path)
    df = df[features + [target]].dropna()
    
    # Normalizar (Min-Max) qualquer coluna que tenha valores fora de [0, 1]
    for col in features:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min < 0 or col_max > 1:
            df[col] = (df[col] - col_min) / (col_max - col_min)
            
    with open(output_path, 'w') as f:
        f.write("@relation Dataset_Filtrado\n")
        
        for col in features:
            f.write(f"@attribute {col} real[-0.01,1.01]\n")
            
        classes = ",".join(str(int(c)) for c in sorted(df[target].unique()))
        f.write(f"@attribute {target} {{{classes}}}\n")
        
        # IDÊNTICO AO ECOLI: sem espaços na lista de features e @output no singular
        features_str = ", ".join(features)
        f.write(f"@inputs {features_str}\n")
        f.write(f"@output {target}\n")
        f.write("@data\n")
        
        # Escrevendo os dados linha por linha
        for _, row in df.iterrows():
            # A vírgula colada no número, igual ao ecoli
            row_str = ", ".join(f"{row[col]:.5f}" for col in features)
            row_str += f", {int(row[target])}\n"
            f.write(row_str)

output_file = 'dataset_perfeito_keel.dat'
create_keel_dat('dataset.csv', output_file)
print(f"Arquivo '{output_file}' gerado com a formatação estrita do KEEL!")
