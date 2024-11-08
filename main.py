import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv"
dados_carros = pd.read_csv(url)

print("Colunas disponíveis:")
print(dados_carros.columns)

atributos_selecionados = ['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
                          'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm',
                          'citympg', 'highwaympg', 'fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginetype']
X = dados_carros[atributos_selecionados]
y = dados_carros['price']
# ======================================================================================
# Convertemos variáveis categóricas em numéricas e normalizamos todas as características e a variável alvo.
le = LabelEncoder()
categorical_columns = ['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginetype']
for col in categorical_columns:
    X[col] = le.fit_transform(X[col])

scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()


# Convertemos variáveis categóricas em numéricas e normalizamos todas as características e a variável alvo.
# ======================================================================================
# Estas são as funções de ativação que serão usadas na rede neural e suas respectivas derivadas.
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def derivada_sigmoid(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def derivada_tanh(x):
    return 1 - x ** 2


def relu(x):
    return np.maximum(0, x)


def derivada_relu(x):
    return np.where(x > 0, 1, 0)


# Estas são as funções de ativação que serão usadas na rede neural e suas respectivas derivadas.
# ======================================================================================
# inicialização da rede neural. Definimos a arquitetura, funções de ativação e inicializamos os pesos usando a inicialização He.
# metodos da classe MLP:
# propagar_adiante: Realiza a propagação direta na rede.
# retropropagar: Implementa o algoritmo de retropropagação com otimizador Adam.
# treinar: Treina a rede por um número especificado de épocas.
# prever: Faz previsões usando a rede treinada.
class MLP:
    def __init__(self, tamanhos_camadas, funcao_ativacao, derivada_ativacao):
        self.tamanhos_camadas = tamanhos_camadas
        self.funcao_ativacao = funcao_ativacao
        self.derivada_ativacao = derivada_ativacao
        self.pesos = [np.random.randn(y, x + 1) * np.sqrt(2. / (x + y)) for x, y in
                      zip(tamanhos_camadas[:-1], tamanhos_camadas[1:])]
        self.m = [np.zeros_like(w) for w in self.pesos]
        self.v = [np.zeros_like(w) for w in self.pesos]

    # inicialização da rede neural. Definimos a arquitetura, funções de ativação e inicializamos os pesos usando a inicialização He.
    # metodos da classe MLP:
    # propagar_adiante: Realiza a propagação direta na rede.
    # retropropagar: Implementa o algoritmo de retropropagação com otimizador Adam.
    # treinar: Treina a rede por um número especificado de épocas.
    # prever: Faz previsões usando a rede treinada.
    # ======================================================================================
    # esta função realiza a propagação para frente na rede neural:

    #   Começa com as entradas X e as armazena em uma lista de ativações.
    #   Para cada camada (representada pelos pesos w):
    #       Adiciona um bias (1) à entrada.
    #       Multiplica a entrada pela matriz de pesos transposta.
    #       Aplica a função de ativação ao resultado.
    #       Armazena o resultado na lista de ativações.
    #   Retorna todas as ativações (útil para o processo de retropropagação).
    def propagar_adiante(self, X):
        ativacoes = [X]
        for w in self.pesos:
            X = np.insert(X, 0, 1, axis=1)
            X = self.funcao_ativacao(X.dot(w.T))
            ativacoes.append(X)
        return ativacoes

    #   Começa com as entradas X e as armazena em uma lista de ativações.
    #   Para cada camada (representada pelos pesos w):
    #       Adiciona um bias (1) à entrada.
    #       Multiplica a entrada pela matriz de pesos transposta.
    #       Aplica a função de ativação ao resultado.
    #       Armazena o resultado na lista de ativações.
    #   Retorna todas as ativações (útil para o processo de retropropagação).
    # ======================================================================================
    #   Esta função implementa o algoritmo de retropropagação com o otimizador Adam:
    #
    #    Calcula o erro na camada de saída.
    #    Propaga o erro de volta através da rede, calculando os deltas para cada camada.
    #    Usa o otimizador Adam para atualizar os pesos:
    #        Calcula os gradientes.
    #        Atualiza as médias móveis do primeiro (m) e segundo (v) momento.
    #        Calcula as versões corrigidas de m e v.
    #        Atualiza os pesos usando a fórmula do Adam.
    def retropropagar(self, X, y, ativacoes, taxa_aprendizado, t):
        erro = y.reshape(-1, 1) - ativacoes[-1]
        deltas = [erro * self.derivada_ativacao(ativacoes[-1])]

        for i in reversed(range(len(self.pesos) - 1)):
            delta = deltas[-1].dot(self.pesos[i + 1][:, 1:]) * self.derivada_ativacao(ativacoes[i + 1])
            deltas.append(delta)

        deltas.reverse()

        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for i in range(len(self.pesos)):
            camada = np.insert(ativacoes[i], 0, 1, axis=1)
            grad = deltas[i].T.dot(camada)
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grad ** 2)
            m_hat = self.m[i] / (1 - beta1 ** t)
            v_hat = self.v[i] / (1 - beta2 ** t)
            self.pesos[i] += taxa_aprendizado * m_hat / (np.sqrt(v_hat) + epsilon)

    # Esta função implementa o algoritmo de retropropagação com o otimizador Adam:
    #
    #    Calcula o erro na camada de saída.
    #    Propaga o erro de volta através da rede, calculando os deltas para cada camada.
    #    Usa o otimizador Adam para atualizar os pesos:
    #        Calcula os gradientes.
    #        Atualiza as médias móveis do primeiro (m) e segundo (v) momento.
    #        Calcula as versões corrigidas de m e v.
    #        Atualiza os pesos usando a fórmula do Adam.
    # ======================================================================================
    # Esta função treina a rede neural:
    #
    #    Itera sobre o número especificado de épocas.
    #    Em cada época, divide os dados em lotes.
    #    Para cada lote:
    #        Realiza a propagação para frente.
    #        Realiza a retropropagação e atualiza os pesos.
    #    Após cada época, calcula e armazena o erro quadrático médio.
    #    A cada 100 épocas, imprime o erro atual.
    #    Retorna a lista de erros ao longo do treinamento.

    def treinar(self, X, y, epocas, taxa_aprendizado, tamanho_lote):
        erros = []
        t = 0
        for epoca in range(epocas):
            for i in range(0, len(X), tamanho_lote):
                t += 1
                lote_X = X[i:i + tamanho_lote]
                lote_y = y[i:i + tamanho_lote]
                ativacoes = self.propagar_adiante(lote_X)
                self.retropropagar(lote_X, lote_y, ativacoes, taxa_aprendizado, t)

            previsoes = self.prever(X)
            eqm = np.mean((y - previsoes.flatten()) ** 2)
            erros.append(eqm)

            if epoca % 100 == 0:
                print(f"Época {epoca}: Erro Quadrático Médio = {eqm:.4f}")

        return erros

    # Esta função treina a rede neural:
    #
    #    Itera sobre o número especificado de épocas.
    #    Em cada época, divide os dados em lotes.
    #    Para cada lote:
    #        Realiza a propagação para frente.
    #        Realiza a retropropagação e atualiza os pesos.
    #    Após cada época, calcula e armazena o erro quadrático médio.
    #    A cada 100 épocas, imprime o erro atual.
    #    Retorna a lista de erros ao longo do treinamento.
    # ======================================================================================
    # Esta função faz previsões usando a rede treinada:
    #
    #    Realiza a propagação para frente.
    #    Retorna apenas a ativação da última camada (saída da rede).
    def prever(self, X):
        return self.propagar_adiante(X)[-1]


# Esta função faz previsões usando a rede treinada:
#
#    Realiza a propagação para frente.
#    Retorna apenas a ativação da última camada (saída da rede).
# ======================================================================================
# Esta função cria e treina uma rede neural com configurações específicas e retorna os erros de treinamento e o erro final.
def testar_configuracao(X, y, tamanhos_camadas, funcao_ativacao, derivada_ativacao, epocas, taxa_aprendizado,
                        tamanho_lote, amostras):
    X_treino, X_teste, y_treino, y_teste = train_test_split(X[:amostras], y[:amostras], test_size=0.2, random_state=42)

    mlp = MLP(tamanhos_camadas, funcao_ativacao, derivada_ativacao)
    erros = mlp.treinar(X_treino, y_treino, epocas, taxa_aprendizado, tamanho_lote)

    y_pred = mlp.prever(X_teste)
    eqm = np.mean((y_teste - y_pred.flatten()) ** 2)

    return erros, eqm


# Esta função cria e treina uma rede neural com configurações específicas e retorna os erros de treinamento e o erro final.
# ======================================================================================
# configs
configuracoes = [
    {"camadas": [X.shape[1], 128, 64, 32, 16, 1], "ativacao": (relu, derivada_relu), "taxa": 0.001, "epocas": 1000,
     "tamanho_lote": 32, "amostras": 1000, "nome": "ReLU"},
    {"camadas": [X.shape[1], 128, 64, 32, 16, 1], "ativacao": (tanh, derivada_tanh), "taxa": 0.001, "epocas": 1000,
     "tamanho_lote": 32, "amostras": 1000, "nome": "Tanh"},
    {"camadas": [X.shape[1], 128, 64, 32, 16, 1], "ativacao": (sigmoid, derivada_sigmoid), "taxa": 0.001,
     "epocas": 1000, "tamanho_lote": 32, "amostras": 1000, "nome": "Sigmoid"},
    {"camadas": [X.shape[1], 128, 64, 32, 16, 1], "ativacao": (relu, derivada_relu), "taxa": 0.01, "epocas": 1000,
     "tamanho_lote": 32, "amostras": 1000, "nome": "Alta Taxa de Aprendizado"},
    {"camadas": [X.shape[1], 128, 64, 32, 16, 1], "ativacao": (relu, derivada_relu), "taxa": 0.001, "epocas": 2000,
     "tamanho_lote": 32, "amostras": 1000, "nome": "Mais Épocas"},
    {"camadas": [X.shape[1], 128, 64, 32, 16, 1], "ativacao": (relu, derivada_relu), "taxa": 0.001, "epocas": 1000,
     "tamanho_lote": 32, "amostras": 2000, "nome": "Mais Amostras"}
]

resultados = []
# configs
# ======================================================================================
# este loop treina a rede com cada configuração e armazena os resultados.
for config in configuracoes:
    print(f"\nTreinando com configuração: {config['nome']}")
    erros, eqm = testar_configuracao(X, y, config['camadas'], config['ativacao'][0], config['ativacao'][1],
                                     config['epocas'], config['taxa'], config['tamanho_lote'], config['amostras'])
    resultados.append((config['nome'], erros, eqm))
# este loop treina a rede com cada configuração e armazena os resultados.]
# ======================================================================================
plt.figure(figsize=(20, 15))
for i, (nome, erros, eqm) in enumerate(resultados):
    plt.subplot(3, 2, i + 1)
    plt.plot(erros)
    plt.title(f"{nome}\nEQM Final: {eqm:.4f}")
    plt.xlabel('Época')
    plt.ylabel('Erro Quadrático Médio')
    plt.grid(True)

plt.tight_layout()
plt.show()

nomes = [r[0] for r in resultados]
eqms = [r[2] for r in resultados]

plt.figure(figsize=(12, 6))
plt.bar(nomes, eqms)
plt.title('Comparação de EQM Final')
plt.xlabel('Configuração')
plt.ylabel('EQM')
for i, v in enumerate(eqms):
    plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()













# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Explicação betas e epsilon
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  beta1 (β1):
#      Valor típico: 0.9
#      Este é o fator de decaimento para a estimativa do primeiro momento (média).
#      Controla quanto peso é dado ao gradiente atual vs. gradientes passados.
#      Um valor próximo a 1 significa que a média móvel muda lentamente, dando mais peso aos gradientes passados.
#
#  beta2 (β2):
#      Valor típico: 0.999
#      Este é o fator de decaimento para a estimativa do segundo momento (variância não centralizada).
#      Controla quanto peso é dado ao quadrado do gradiente atual vs. quadrados de gradientes passados.
#      Um valor próximo a 1 significa que a estimativa da variância muda lentamente.
#
#  epsilon (ε):
#      Valor típico: 1e-8 (ou seja, 0.00000001)
#      É um valor pequeno adicionado ao denominador para evitar divisão por zero.
#      Ajuda na estabilidade numérica do algoritmo.
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Explicação camdas
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Esta lista define a estrutura da rede neural, especificando o número de neurônios em cada camada. Vamos analisar cada elemento:
#
#    X.shape[1]: Este é o número de características de entrada. É dinâmico e se ajusta automaticamente ao número de colunas em seus dados de entrada.
#
#    64: Este é o número de neurônios na primeira camada oculta.
#
#    32: Este é o número de neurônios na segunda camada oculta.
#
#    16: Este é o número de neurônios na terceira camada oculta.
#
#    1: Este é o número de neurônios na camada de saída. Neste caso, temos apenas um neurônio porque estamos prevendo um único valor (o preço do carro).
#
# Então, esta configuração específica cria uma rede neural com:
#
#    Uma camada de entrada com tamanho igual ao número de características
#    Três camadas ocultas com 64, 32 e 16 neurônios respectivamente
#    Uma camada de saída com 1 neurônio
#
# A inclusão desta configuração de camadas permite:
#
#    Flexibilidade: Você pode facilmente experimentar diferentes arquiteturas de rede neural alterando estes números.
#
#    Profundidade: Ao usar múltiplas camadas ocultas, você está criando uma "rede neural profunda", que pode aprender representações mais complexas dos dados.
#
#    Ajuste fino: Você pode ajustar o número de neurônios em cada camada para encontrar o equilíbrio ideal entre capacidade de aprendizado e overfitting.
#
#    Comparação: Ao manter esta configuração consistente entre diferentes testes (variando apenas outros parâmetros como função de ativação ou taxa de aprendizado), você pode fazer comparações mais justas.
#
# Esta abordagem de definir a arquitetura da rede através de uma lista de camadas é uma prática comum em implementações de redes neurais, pois oferece grande flexibilidade e facilidade de experimentação com diferentes estruturas de rede.
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=






















