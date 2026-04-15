import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

### Пятая лаба
def plot_trajectory(f,path,xlim=(-15,15),ylim=(-15,15),grid=200):
    '''
    Функция для визуализации адаптивного метода случайного поиска
    '''
    levels=50
    x=np.linspace(xlim[0],xlim[1],grid)
    y=np.linspace(ylim[0],ylim[1],grid)
    X,Y=np.meshgrid(x,y)

    Z=np.zeros_like(X)
    for i in range(grid):
        for j in range(grid):
            Z[i,j]=f(np.array([X[i,j],Y[i,j]]))

    plt.figure(figsize=(8,6))
    plt.contourf(X,Y,Z,levels=levels,cmap='viridis',alpha=0.7)
    plt.plot(path[:,0],path[:,1],'w-')
    plt.scatter(path[:,0],path[:,1],c='red',s=50)
    plt.scatter(path[0,0],path[0,1],c='yellow',s=50)
    plt.grid()
    plt.show()



def plot_rejection_random_search(f,path,xlim=(-15,15),ylim=(-15,15),grid=200):
    '''
    Функция для визуализации метода случайного поиска с возвратом при неудачном шаге
    '''
    levels=50
    x=np.linspace(xlim[0],xlim[1],grid)
    y=np.linspace(ylim[0],ylim[1],grid)
    X,Y=np.meshgrid(x,y)

    Z=np.zeros_like(X)
    for i in range(grid):
        for j in range(grid):
            Z[i,j]=f(np.array([X[i,j],Y[i,j]]))

    plt.figure(figsize=(8,6))
    plt.contourf(X,Y,Z,levels=levels,cmap='viridis',alpha=0.7)
    plt.plot(path[:,0],path[:,1],'w-')
    plt.scatter(path[:,0],path[:,1],c='red',s=50)
    plt.scatter(path[0,0],path[0,1],c='yellow',s=50)
    plt.grid()
    plt.show()