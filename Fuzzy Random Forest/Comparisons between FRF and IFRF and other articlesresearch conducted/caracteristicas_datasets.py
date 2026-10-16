from ucimlrepo import fetch_ucirepo
import pandas as pd

datasets_uci = {
    'Dermatology': 33,
    'Iris': 53,
    'Glass Identification': 42,
    'Breast Cancer Wisconsin (Original)': 15,
    'Statlog (Heart)': 145,
    'Wine': 109,
    'Zoo': 111,
    'Statlog (Australian Credit)': 143,
    'Hepatitis': 46,
    'Ionosphere': 52
}

info_list = []

print("Extraindo metadados da UCI...")
for name, ds_id in datasets_uci.items():
    dataset = fetch_ucirepo(id=ds_id)
    meta = dataset.metadata
    
    try:
        y = dataset.data.targets
        num_classes = y.iloc[:, 0].nunique() if y is not None else "N/A"
    except:
        num_classes = "N/A"

    info_list.append({
        'Dataset': name,
        'Instâncias': meta.num_instances,
        'Atributos (Features)': meta.num_features,
        'Classes': num_classes,
        'Valores Ausentes?': 'Sim' if meta.has_missing_values else 'Não',
        'Tipos de Feature': meta.feature_types
    })

df_info = pd.DataFrame(info_list)

print("Resumo Datasets UCI:")
print(df_info.to_string(index=False))