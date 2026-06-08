import matplotlib.pyplot as plt
import numpy as np

# Configuração dos modelos extraídos
modelos = [
    'CORE-C', 'HIDER-C', 'IVTURS-C', 'NSLV-C', 'SGERD-C'
]

# Acurácias de Treino e Teste extraídas (Convertidas para % e arredondadas)
train_acc = [53.68, 60.76, 61.49, 52.46, 51.76]
test_acc  = [54.11, 59.98, 61.45, 52.15, 51.78]

# Ordem crescente baseado no teste
indices_ordenados = np.argsort(test_acc)

modelos = [modelos[i] for i in indices_ordenados]
train_acc = [train_acc[i] for i in indices_ordenados]
test_acc = [test_acc[i] for i in indices_ordenados]

x = np.arange(len(modelos))
width = 0.35  # Largura das barras

# Figura com proporção ajustada
fig, ax = plt.subplots(figsize=(10, 6))

# Construção das barras 
rects1 = ax.bar(x - width/2, train_acc, width, label='Treino', 
                capsize=5, color='#4C72B0', edgecolor='black')
rects2 = ax.bar(x + width/2, test_acc, width, label='Teste', 
                capsize=5, color='#DD8452', edgecolor='black')

# Personalização para padrão acadêmico
ax.set_ylabel('Acurácia (%)', fontsize=12, fontweight='bold')
ax.set_title('Desempenho Preditivo dos Modelos', fontsize=14, fontweight='bold')
ax.set_xticks(x)

# Rótulos do eixo X
ax.set_xticklabels(modelos, fontsize=11, rotation=0) 

# Limite do eixo Y ajustado para comportar o valor máximo (~61.5%) e suas labels
ax.set_ylim(45, 70) 
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

plt.savefig('grafico_resultados_novos_modelos.png', dpi=300)
plt.show()