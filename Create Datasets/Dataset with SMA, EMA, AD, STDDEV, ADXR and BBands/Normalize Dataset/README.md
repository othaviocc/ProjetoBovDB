# Normalização dos Indicadores do Dataset

O arquivo **dados_indicadores.csv** contém as seguintes colunas:

datetime,id_ticker,open,close,high,low,average,volume,business,amount_stock,date,
SMA_3,EMA_3,SMA_5,EMA_5,SMA_7,EMA_7,SMA_9,EMA_9,SMA_11,EMA_11,
std_close3,std_open3,std_close5,std_open5,std_close7,std_open7,std_close9,std_open9,std_close11,std_open11,
Bollinger_Mid,Bollinger_Upper,Bollinger_Lower,AD_Line,ADXR

A partir dele, os indicadores técnicos foram normalizados na escala **0 a 1**, o que facilita comparações e melhora a performance de modelos de machine learning.  

A normalização foi dividida em **duas etapas**:

1. **normalizando_passo1.py** → Primeira parte do cálculo das médias móveis (SMA e EMA) e desvio padrão (std).  
2. **normalizando_passo2.py** → Finalização da normalização aplicando o restante da fórmula e normalização de Bollinger, ADXR, criação da coluna de tendência (`trend`), mantendo o AD_Line sem alteração.

---

## Fórmulas de Normalização

### SMA / EMA
\[
\text{SMA/EMA}_{\text{norm}} = \frac{\text{Indicador} - \min(\text{Indicador})}{\max(\text{Indicador}) - \min(\text{Indicador})}
\]

### STD (std_close / std_open)
\[
\text{STD}_{\text{norm}} = \frac{\text{STD} - \min(\text{STD})}{\max(\text{STD}) - \min(\text{STD})}
\]

### Bollinger Bands
\[
\text{Bollinger\_Norm} = \frac{\text{Close}_{t-1} - \text{Bollinger\_Lower}}{\text{Bollinger\_Upper} - \text{Bollinger\_Lower}}
\]

### ADXR
\[
\text{ADXR}_{\text{norm}} = \frac{\text{ADXR}}{100}
\]

### AD_Line
\[
AD\_Line \text{ (mantido sem alteração)}
\]

### Trend
\[
\text{trend} =
\begin{cases}
1, & \text{se Close}_t > \text{Close}_{t-1} \\
0, & \text{se Close}_t \le \text{Close}_{t-1}
\end{cases}
\]
