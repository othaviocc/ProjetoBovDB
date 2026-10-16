# Processamento de Dados de 1 Minuto - Mini Índice (WIN)

Este repositório contém os scripts para geração, limpeza e estruturação da base de dados de 1 minuto dos contratos futuros de Mini Índice, extraídos do ambiente BovDB. O dataset resultante é otimizado para o cálculo de indicadores técnicos e treinamento de modelos preditivos em séries temporais financeiras.

## 1. Geração do Dataset Base

O arquivo bruto é gerado executando o script `dados.py`. Ele consulta o banco de dados e cria o arquivo `dados.csv`, totalizando **73.887 linhas** de histórico minuto a minuto.

O dataset contém as seguintes variáveis (colunas):
`id_ticker`, `date`, `time`, `open`, `close`, `high`, `low`, `average`, `volume`, `business`, `amount_stock`

> **Aplicação:** A partir desta estrutura, a pipeline permite a extração de indicadores técnicos de mercado que atuam como *features*, além da criação da coluna `trend`, que é isolada como a variável alvo (alvo de predição) para a modelagem.

## 2. Tratamento e Limpeza de Anomalias

Durante a exploração dos dados, foram identificados registros com erros de formatação nos horários de negociação (ex: horas registradas no padrão `9x:xx:xx` no final da tarde). 

Para corrigir a integridade temporal da base, o script `DropSamples_dados.csv.py` foi executado. 
* **Total de amostras anômalas removidas:** 20.064 linhas.
* **Resultado:** Um dataset limpo, contínuo e sem saltos irregulares de horário.


## 3. Gerar dataset com dados de 1, 5, 10 e 15 minutos




## 4. Gerar Indicadores Tecnicos

`indicadores.py`


## 5. Gerar Coluna Trend

`ColunaPredição.py`


## 6. Informações Gerais dos dados

Para verificar a distribuição dos dados e entender a rolagem de liquidez, executa-se o script `General_Informations.py`. O histórico abrange o primeiro semestre de 2026, focado unicamente no contrato com maior volume de cada mês:

| ID Ticker | Contrato | Vencimento | Período Ativo na Base | Detalhes da Amostra |
| :---: | :---: | :---: | :--- | :--- |
| **`58431`** | **WING26** | Fev/2026 | `02/01/2026` a `18/02/2026` | O período de captura se encerra exatamente na quarta-feira de vencimento do contrato. |
| **`58432`** | **WINJ26** | Abr/2026 | `02/03/2026` a `31/03/2026` | Representa o contrato de transição com a maior liquidez e volume negociado durante março de 2026. |
| **`58433`** | **WINM26** | Jun/2026 | `01/04/2026` a `17/06/2026` | O período de captura se encerra exatamente na quarta-feira de vencimento do contrato. |
