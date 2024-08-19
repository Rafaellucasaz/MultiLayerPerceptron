import numpy as np;
import random 


def sigmoide(x):
    return 1/(1+ np.exp(-x))

def derivadaSigmoide(x):
    return x * (1 - x)

def inicializarPesos(size1,size2):
    return  np.random.rand(size1,size2) * 2-1



def propagacaoDireta(entrada,pesosEntradaOculta,pesosSaidaOculta) : 
    netsOculta = np.dot(entrada,pesosEntradaOculta)
    ocultaOutput = sigmoide(netsOculta)

    netsSaida = np.dot(ocultaOutput,pesosSaidaOculta)
    saidaOutput = sigmoide(netsSaida)
    
    return ocultaOutput,saidaOutput


def retroPropagacao(entrada,saida,ocultaOutput,saidaOutput,pesosSaidaOculta,pesosEntradaOculta,taxaAprendizado):
    erroSaida = (saida - saidaOutput) * derivadaSigmoide(saidaOutput)
    
    erroOculta = derivadaSigmoide(ocultaOutput) * np.dot(erroSaida,pesosSaidaOculta.T)

    pesosSaidaOculta += np.dot(ocultaOutput.T,erroSaida) * taxaAprendizado
    pesosEntradaOculta += np.dot(entrada.T,erroOculta) * taxaAprendizado

    erroGeral = 0.5 *  np.sum(erroSaida ** 2)
    return pesosSaidaOculta, pesosEntradaOculta, erroGeral




def treino(x,y,taxaAprendizado):
    tamanhoEntrada = x.shape[1]
    tamanhoOculta = 4
    tamanhoSaida = y.shape[1]

    pesosEntradaOculta = inicializarPesos(2,4)
    pesosSaidaOculta = inicializarPesos(4,1)
    erroGeral = 1
    i = 0
    while(erroGeral > 0):
        i = i + 1
        ocultaOutput,saidaOutput = propagacaoDireta(x,pesosEntradaOculta,pesosSaidaOculta)
        pesosSaidaOculta,pesosEntradaOculta,erroGeral = retroPropagacao(x,y,ocultaOutput,saidaOutput,pesosSaidaOculta,pesosEntradaOculta,taxaAprendizado)
        
        if i % 1000 == 0:
            erro = erroGeral
            print(f'Época {i}, Erro {erro}')
    return pesosSaidaOculta, pesosEntradaOculta
    


entrada = np.array([[1,0],[1,1],[0,0],[0,1]])

saida = np.array([[1], [0], [0], [1]])

pesosSaidaOculta,pesosEntradaOculta = treino(entrada,saida,0.9)

ocultaOutput,saidaOutput = propagacaoDireta(entrada,pesosEntradaOculta,pesosSaidaOculta)

print("resultado final:")
print(saidaOutput)

   



