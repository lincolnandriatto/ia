
#%%

import numpy as np
import matplotlib.pyplot as plt

# Classe representando cada camanda da MLP
class Camada:
    def __init__(self, n_entradas, n_neuronios, tipo_funcao_ativacao=None, pesos=None, bias=None, atrazo=None):
        self.pesos = pesos if pesos is not None else np.random.rand(n_entradas, n_neuronios) # Se não passar Pesos, inicia randomicamente
        self.tipo_funcao_ativacao = tipo_funcao_ativacao
        self.bias = bias if bias is not None else np.random.rand(n_neuronios) # Se não passar Bias, inicia randomicamente
        self.atraso = atrazo

    def funcao_soma(self, x):
        r = np.dot(x, self.pesos) + self.bias
        self.valor_ultima_funcao_ativacao = self.funcao_ativacao(r)
        if self.atraso :
            r = r + self.valor_ultima_funcao_ativacao
            self.valor_ultima_funcao_ativacao = self.funcao_ativacao(r)

        return self.valor_ultima_funcao_ativacao


    def funcao_ativacao(self, r):
        # Se nenhuma funca de ativacao for escolhida
        if self.tipo_funcao_ativacao is None:
            return r

        # tanh
        if self.tipo_funcao_ativacao == 'tanh':
            return np.tanh(r)

        # sigmoid
        if self.tipo_funcao_ativacao == 'sigmoid':
            return 1 / (1 + np.exp(-r))

        return r


    def funcao_ativacao_derivada(self, r):
        if self.tipo_funcao_ativacao is None:
            return r

        if self.tipo_funcao_ativacao == 'tanh':
            return 1 - r ** 2

        if self.tipo_funcao_ativacao == 'sigmoid':
            return r * (1 - r)

        return r


class NeuralNetwork:
    def __init__(self, normalizar_serie=False):
        self._camadas = []
        self.normalizar_serie = normalizar_serie

    def adicionar_camada(self, camada):
        self._camadas.append(camada)


    def feed_forward(self, X):
        for camada in self._camadas:
            X = camada.funcao_soma(X)

        return X

    def backpropagation(self, X, y, taxa_aprendizagem):
        # Saida do Feed forward
        valor_saida = self.feed_forward(X)

        # Loop nas camadas anteriores, partindo da de saida para a primeira
        for i in reversed(range(len(self._camadas))):
            camada = self._camadas[i]

            # verifica se é a camada de saída
            if camada == self._camadas[-1]:
                camada.error = y - valor_saida
                # A saída é igual a camada.valor_ultima_funcao_ativacao neste caso
                camada.delta = camada.error * camada.funcao_ativacao_derivada(valor_saida)
            else:
                proxima_camada = self._camadas[i + 1]
                camada.error = np.dot(proxima_camada.delta, proxima_camada.pesos.T) # Alterado AQUI
                camada.delta = camada.error * camada.funcao_ativacao_derivada(camada.valor_ultima_funcao_ativacao)

        # Atualiza os pesos
        for i in range(len(self._camadas)):
            camada = self._camadas[i]
            #  A entrada é ou a saída da camada anterior ou o próprio X(para a primeira camada oculta)
            entrada_a_usar = np.atleast_2d(X if i == 0 else self._camadas[i - 1].valor_ultima_funcao_ativacao)
            camada.pesos += np.dot(entrada_a_usar.T, camada.delta) * taxa_aprendizagem  # Alterado AQUI

    def treinar(self, X, Y, taxa_aprendizagem, maximo_epocas):

        mses = []
        for i in range(maximo_epocas):
            self.backpropagation(X, Y, taxa_aprendizagem) # Alterado AQUI
            if i % 10 == 0:
                mse = np.mean(np.square(Y - nn.feed_forward(X)))
                mses.append(mse)
                print('Época: #%s, MSE: %f' % (i, float(mse)))

        return mses


    def predizer(self, X):
        resultado = self.feed_forward(X)
        return resultado

# Cria a classe de NN
nn = NeuralNetwork()

# Dados de Teste
X = np.array([ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ], [ 1, 1 ] ] )
Y = np.array([ [ 0 ], [ 1 ], [ 1 ], [ 0 ] ] )

# Definir camadas da Rede
nn.adicionar_camada(Camada(len(X[0]), 5, None, None, None, True)) # Linear
nn.adicionar_camada(Camada(5, 10, 'sigmoid', None, None, None)) # Sigmoid
nn.adicionar_camada(Camada(10, 1, 'sigmoid')) # Sigmoid

# Treinar a rede neural
erros = nn.treinar(X, Y, 1, 5000) # Tx Aprendizagem 1 (muito alta) / 5000 épocas

# Testar a NN
print ("Predição 1: ")
entrada = X[0]
esperada = Y[0]
saida_rede = nn.predizer(entrada)
print (str(esperada) + " -> "+ str(saida_rede))

entrada = X[1]
esperada = Y[1]
print ("Predição 2: ")
saida_rede = nn.predizer(entrada)
print (str(esperada) + " -> "+ str(saida_rede))

entrada = X[2]
esperada = Y[2]
print ("Predição 3: ")
saida_rede = nn.predizer(entrada)
print (str(esperada) + " -> "+ str(saida_rede))


entrada = X[3]
esperada = Y[3]
print ("Predição 4: ")
saida_rede = nn.predizer(entrada)
print (str(esperada) + " -> "+ str(saida_rede))


# Plotar o grafico de MSE
plt.plot(erros)
plt.title('Evolução MSE')
plt.xlabel('Epoca (a cada 10)')
plt.ylabel('MSE')
plt.show()
