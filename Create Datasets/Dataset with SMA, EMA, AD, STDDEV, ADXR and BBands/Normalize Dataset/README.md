# Normalização dos Indicadores Técnicos

Este projeto tem como objetivo a normalização dos indicadores técnicos utilizados na análise de séries temporais financeiras. A normalização é realizada em duas etapas:

## Etapa 1: normalizando_passo1.py

- **Objetivo:** Calcular as Médias Móveis Simples (SMA), Médias Móveis Exponenciais (EMA) e Desvios Padrão (STD) para diferentes períodos.
- **Saída:** Geração do arquivo `normalizados_passo1.csv` contendo os indicadores calculados.

## Etapa 2: normalizando_passo2.py

- **Objetivo:** Continuar o processo de normalização aplicando as fórmulas específicas para cada indicador.
- **Saída:** Geração do arquivo `normalizados_passo2.csv` com os indicadores normalizados.

## Estrutura dos Arquivos

- `dados_indicadores.csv`: Contém os dados brutos com os indicadores técnicos.
- `normalizados_passo1.csv`: Contém os dados após a primeira etapa de normalização.
- `normalizados_passo2.csv`: Contém os dados após a segunda etapa de normalização.

## Fórmulas de Normalização

### SMA / EMA

```math
\text{Indicador Normalizado} = \frac{\text{Indicador} - \min(\text{Indicador})}{\max(\text{Indicador}) - \min(\text{Indicador})}
```

### STD (std_close / std_open)

```math
\text{STD Normalizado} = \frac{\text{STD} - \min(\text{STD})}{\max(\text{STD}) - \min(\text{STD})}
```

### Bollinger Bands

```math
\text{Bollinger Normalizado} = \frac{\text{Close}_{t-1} - \text{Bollinger Lower}}{\text{Bollinger Upper} - \text{Bollinger Lower}}
```

### ADXR

```math
\text{ADXR Normalizado} = \frac{\text{ADXR}}{100}
```

### AD Line



