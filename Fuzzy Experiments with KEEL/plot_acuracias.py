import matplotlib.pyplot as plt
import numpy as np

# Configuração dos modelos extraídos do KEEL + RF + XGBoost + Fuzzy RF
modelos = [
    'Chi-RW', 'Fuzzy KNN', 'FH-GBML', 'FURIA', 'Fuzzy FARCHD', 
    'Random Forest', 'XGBoost', 'Fuzzy RF'
]

# Acurácias de Treino e Teste (Convertidas para %)
train_acc = [51.76, 54.93, 61.09, 61.81, 62.89, 64.92, 63.82, 71.32]
test_acc  = [51.71, 54.71, 61.04, 61.61, 62.54, 60.90, 60.34, 61.88]

# ordem crescente baseado no teste
indices_ordenados = np.argsort(test_acc)

modelos = [modelos[i] for i in indices_ordenados]
train_acc = [train_acc[i] for i in indices_ordenados]
test_acc = [test_acc[i] for i in indices_ordenados]


x = np.arange(len(modelos))
width = 0.35  # Largura das barras

# Aumentei a largura da figura para comportar os 8 modelos confortavelmente
fig, ax = plt.subplots(figsize=(12, 6))

# Construção das barras 
rects1 = ax.bar(x - width/2, train_acc, width, label='Treino', 
                capsize=5, color='#4C72B0', edgecolor='black')
rects2 = ax.bar(x + width/2, test_acc, width, label='Teste', 
                capsize=5, color='#DD8452', edgecolor='black')

# Personalização para padrão acadêmico
ax.set_ylabel('Acurácia (%)', fontsize=12, fontweight='bold')
ax.set_title('Desempenho Preditivo: Algoritmos Fuzzy vs Baseados em Árvore', fontsize=14, fontweight='bold')
ax.set_xticks(x)
# Rotacionando levemente para não encavalar os textos
ax.set_xticklabels(modelos, fontsize=11, rotation=15) 
# Limite do eixo Y ajustado para comportar o valor de 71.32% e suas labels
ax.set_ylim(45, 78) 
ax.legend(fontsize=12, loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Função para adicionar o texto dos valores em cima das barras
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # Deslocamento vertical de 4 pontos
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.savefig('grafico_resultados_fuzzy_trees.png', dpi=300)
plt.show()
