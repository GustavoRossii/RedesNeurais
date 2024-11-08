# Rede Neural para Previsão de Preços de Carros

Este projeto implementa uma rede neural multicamadas (MLP) para prever preços de carros usando várias características do veículo. O modelo é implementado do zero em Python, utilizando apenas NumPy para operações matemáticas.

## Integrantes do Grupo

- Gustavo Rossi
- Bruno Trevizan
- Yuji Kiyota

## Visão Geral

O projeto consiste em uma implementação de uma rede neural feedforward com as seguintes características:

- Utiliza o algoritmo de otimização Adam para ajuste de pesos
- Suporta várias funções de ativação (ReLU, Sigmoid, Tanh)
- Permite configurações flexíveis de camadas e hiperparâmetros
- Inclui visualizações de desempenho para diferentes configurações

## Estrutura do Código

O código está organizado da seguinte forma:

1. Carregamento e pré-processamento de dados
2. Implementação da classe MLP (Multi-Layer Perceptron)
3. Funções de ativação e suas derivadas
4. Funções de treinamento e avaliação
5. Configurações de teste para diferentes arquiteturas e hiperparâmetros
6. Visualização de resultados

## Como Usar

1. Certifique-se de ter as bibliotecas necessárias instaladas:

pip install numpy pandas matplotlib scikit-learn

3. O script irá treinar várias configurações de rede neural e exibir gráficos comparativos do desempenho.

## Resultados

O script gera dois tipos de visualizações:

1. Gráficos de erro ao longo do tempo para cada configuração testada
2. Um gráfico de barras comparando o Erro Quadrático Médio (EQM) final de todas as configurações

Estes gráficos ajudam a identificar qual configuração de rede neural teve o melhor desempenho para o conjunto de dados de preços de carros.

## Notas Adicionais

- O dataset utilizado é carregado diretamente de uma URL e contém informações sobre diversos aspectos de carros, incluindo suas características físicas e mecânicas.
- As configurações de teste incluem variações na arquitetura da rede, funções de ativação e hiperparâmetros como taxa de aprendizado e número de épocas.
- Este projeto foi desenvolvido como parte de um estudo acadêmico sobre redes neurais e machine learning.
