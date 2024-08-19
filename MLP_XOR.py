import numpy as np;
 

def sigmoide(x):
    #funcao sigmoide
    return 1/(1+ np.exp(-x))

def derivadaSigmoide(x):
    #derivada da sigmoide
    return x * (1 - x)

def inicializarPesos(size1,size2):
    #funcao para inicializar pesos com valores aleatorios
    return  np.random.rand(size1,size2) * 2-1



def propagacaoDireta(entrada,pesosEntradaOculta,pesosSaidaOculta) : 
    #calculando Nets da camada oculta
    netsOculta = np.dot(entrada,pesosEntradaOculta)
    #aplicando funcao de ativacao sigmoide
    ocultaOutput = sigmoide(netsOculta)
    #calculando Nets da camada de saida
    netsSaida = np.dot(ocultaOutput,pesosSaidaOculta)
    #aplicando funcao de ativacao sigmoide
    saidaOutput = sigmoide(netsSaida)
    
    return ocultaOutput,saidaOutput

def testeFinal(entrada,pesosEntradaOculta,PesosSaidaOculta):
     #calculando Nets da camada oculta
    netsOculta = np.dot(entrada,pesosEntradaOculta)
    #aplicando funcao de ativacao sigmoide
    ocultaOutput = sigmoide(netsOculta)
    #calculando Nets da camada de saida
    netsSaida = np.dot(ocultaOutput,pesosSaidaOculta)
    #aplicando funcao de ativacao sigmoide
    saidaOutput = sigmoide(netsSaida)
    return saidaOutput


def retroPropagacao(entrada,saida,ocultaOutput,saidaOutput,pesosSaidaOculta,pesosEntradaOculta,taxaAprendizado):
    #calculando erro da camada de saida
    erroSaida = (saida - saidaOutput) * derivadaSigmoide(saidaOutput)
    #calculando erro da camada oculta
    erroOculta = derivadaSigmoide(ocultaOutput) * np.dot(erroSaida,pesosSaidaOculta.T)

    #ajustando pesos
    pesosSaidaOculta += np.dot(ocultaOutput.T,erroSaida) * taxaAprendizado
    pesosEntradaOculta += np.dot(entrada.T,erroOculta) * taxaAprendizado
    #calculando erro da rede
    erroGeral = 0.5 * np.sum(erroSaida ** 2)
    return pesosSaidaOculta, pesosEntradaOculta, erroGeral



#treino da rede
def treino(x,y,taxaAprendizado):

    neuroniosEntrada = 2
    neuroniosOculta = 4
    neuroniosSaida = 1
    #inicializa pesos
    pesosEntradaOculta = inicializarPesos(neuroniosEntrada,neuroniosOculta)
    pesosSaidaOculta = inicializarPesos(neuroniosOculta,neuroniosSaida)
    erroGeral = 1
    i = 0
    while(erroGeral > 0.000001):
        i += 1
        #propagacao direta
        ocultaOutput,saidaOutput = propagacaoDireta(x,pesosEntradaOculta,pesosSaidaOculta)
        #retropropagacao
        pesosSaidaOculta,pesosEntradaOculta,erroGeral = retroPropagacao(x,y,ocultaOutput,saidaOutput,pesosSaidaOculta,pesosEntradaOculta,taxaAprendizado)
        
        if i % 1000 == 0:
            
            print(f'Época {i}, Erro {erroGeral}')
    return pesosSaidaOculta, pesosEntradaOculta
    


entrada = np.array([[1,0],[1,1],[0,0],[0,1]])

saida = np.array([[1], [0], [0], [1]])

#treina rede
pesosSaidaOculta,pesosEntradaOculta = treino(entrada,saida,0.1)
#testa rede neural depois do treino 
saidaOutput = testeFinal(entrada,pesosEntradaOculta,pesosSaidaOculta)

print("resultado final:")
print(saidaOutput)