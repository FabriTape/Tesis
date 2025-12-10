#!/home/tutturuu/spyder-env/bin python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt 
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t
## Definimos el modelo que queremos ajustar
def modelo(x,a,b,c,d):
    return a*(x-d)*np.exp(-b*x)+c
## Le sacamos las derivadas aprciales respecto  a los parámetros que le queremos ajustar
def derA(x,a,b,c,d):
    return (x-d)*np.exp(-b*x)
def derB(x,a,b,c,d):
    return -a*(x-d)*x*np.exp(-b*x)
def derC(x,a,b,c,d):
    return 1
def derD(x,a,b,c,d):
    return a*(b*x-1)*np.exp(-b*x)
## Definimos el gradiente 
def grad(x,popt):
    return np.array([derA(x,*popt),derB(x,*popt),derC(x,*popt),derD(x,*popt)])
## Buscamos la desviación estandar del ajuste usando la matriz de covarianza
def errorf(absisa,popt,pcov):
    return np.array([np.sqrt(grad(x,popt).T @ pcov @ grad(x,popt)) if x!=0 else 0 for x in absisa])
## Energía de las partículas de la cama menos el proyectil a t=0
def ajustes_energia(Data_collect:dict,paso,u=5):
    K = np.array(Data_collect["K"])
    K_eyectados = np.array(Data_collect['Eyectados_K'])
    K_cama = K-K_eyectados
    ## Hacemos el ajuste

    X_red = np.linspace(0,len(K)-1,len(K))
    popt, pcov = curve_fit(modelo,X_red,K_cama)
    
    ## Velocidad del CM a tiempo t
    Vcm = np.array(Data_collect['Vcm'])
    Vcm2 = np.linalg.norm(Vcm,axis=1)
    ## Para hacer el plot continuo
    X_cont = np.linspace(0,len(K)-1,500)
    t_student = t.ppf(0.975,len(K_cama))
    
    #########################################
    # Plots #################################
    #########################################
    print(f"Delta K de la cama entre el principio y el final de la simulación\n        -{'Aumentó:' if (K_cama[-1]-K_cama[0])>0 else 'Disminuyó:'} {(K_cama[-1]-K_cama[0])/K_cama[0]*100:.2f} %")
    fig,ax = plt.subplots(2,1,figsize=(6,6))
    Y = modelo(X_cont,*popt)
    E = errorf(X_cont,popt,pcov)
    # En este plot tiene la energía cinética de la cama, con ajuste 
    ax[0].plot(X_red,K_cama,"r-",label="Energía cinética de la cama")
    ax[0].plot(X_cont,Y,"b-",label="Modelo")
    ax[0].fill_between(X_cont,Y-E*t_student,Y+E*t_student,alpha=0.5,label=r"error del ajuste")
    ax[0].plot(Vcm2/max(Vcm2),"o",label="Módulo del momento del sistema\nnormalizado",markersize=1)
    ax[0].legend()
    ax[0].set_xlabel(f"Paso de tiempo {paso}")
    ax[0].set_ylabel("Energía cinética K/m")
    
    ax[1].plot(Data_collect['Eyección'],"o",label="Parículas eyectadas")
    ax[1].set_xlabel("Paso de tiempo")
    ax[1].set_ylabel("Cantidad de parículas eyectadas")
    ax[1].legend()
    plt.show()
    
    plt.plot(K_cama,"g--",label="Energía cinética de la cama y eyectados")
    plt.plot(K_eyectados,"r--",label="Energía cinética de los eyectados")
    plt.plot(K,"b--",label="Energía cinética de reposo")
    #plt.fill_between(np.linspace(0,1000,1001),np.array(Data_collect["Min"])*0.5,np.array(Data_collect["Max"])*0.5,alpha=0.5,label=r"Máx y min de la cama")
    
    plt.xlim((0,26))
    plt.ylim((0,4.5))
    plt.legend()
