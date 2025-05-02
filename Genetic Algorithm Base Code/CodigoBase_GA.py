''' Este codigo serve de auxilio para a utilização de GA utilizando a biblioteca DEAP (https://github.com/deap/deap)
eu utilizo esta base para encontrar os melhores parametros de indicadores tecnicos, entre outros...
 '''
from deap import base, creator, tools, algorithms
import random
# outros muito utilizados:
import numpy as np
import pandas as pd

# importar dados a serem utilizados, ajuste da maneira que for utilizar
df = 'banco_de_dados.csv'

# criação do GA
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# o numero de bits necessarios para cada parametro
n_bits_1 = 0        
n_bits_2 = 0
n_bits_3 = 0
n_bits_4 = 0
n_bits_5 = 0

total_bits = n_bits_1 + n_bits_2 + n_bits_3 + n_bits_4 + n_bits_5

toolbox.register("attr_bin", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bin, total_bits)   
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def calculate_fitness(df, p1, p2, ):
    # calcula a fitness do teu problema
    resultado = ...  # algum cálculo
    return (resultado,)  


# decodifica um gene representado por uma sequência binária para um valor real dentro de um intervalo definido
def decode_binary(gene, minimo, maximo, n_bits):
    binary_str = ''.join(map(str, gene))
    int_value = int(binary_str, 2)
    max_value = 2 ** n_bits - 1
    return minimo + (int_value / max_value) * (maximo - minimo)

def evaluate(individual):
    # Decodifica os bits do indivíduo para obter os parâmetros reais
    # Usa esses parâmetros para calcular e retornar a fitness da solução
    start = 0
    parametro1 = int(decode_binary(individual[start:start + n_bits_1], 2, 30, n_bits_1))
    start += n_bits_1
    parametro2 = decode_binary(individual[start:start + n_bits_2], 0.6, 4, n_bits_2)
    '''
    parametro3 = ...
    parametro4 = ... 
    parametro5 = ...
    ''' 
    return calculate_fitness(df, parametro1, parametro2, ...)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb = 0.05)    #exemplo de mutação
toolbox.register("select", tools.selTournament, tournsize = 3)   #exemplo torneio

def main(df, n_gen, pop_size):

    # Aqui na função main é onde tudo acontece
    
    # Primeiro, criamos a população inicial com indivíduos aleatórios
    # Cada indivíduo é uma lista de bits que representa possíveis soluções
    
    # Em seguida, criamos o Hall of Fame (hof), que armazena o melhor indivíduo encontrado durante a execução
    
    # Também definimos as estatísticas que queremos acompanhar durante as gerações:
    # mínimo (min), média (mean) e máximo (max) dos valores de fitness
    
    # Depois disso, usamos o algoritmo genético simples (eaSimple) fornecido pela DEAP
    # Ele realiza o loop evolutivo: seleção, cruzamento, mutação e avaliação ao longo das gerações
    # cxpb é a probabilidade de cruzamento e mutpb é a de mutação
    
    # Ao final da execução, o melhor indivíduo encontrado é armazenado no Hall of Fame (hof[0])
    # Podemos então decodificar esse indivíduo e ver quais foram os melhores parâmetros
    
    # Por fim, retornamos esse melhor indivíduo, que teve o maior valor de fitness durante a evolução

    pass

#para saber mais sobre o que o DEAP possue entre em: https://deap.readthedocs.io/en/master/# 