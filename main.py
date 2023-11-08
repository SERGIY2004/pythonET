import random
import math
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt

N = 5 # int(input('Number of coordinates = '))
xmin = -10  # float(input('Mininum range value = '))
xmax = 10  # float(input('Maximum range value = '))
mu = 10  # int(input('Number of parents in each age is '))
lam = 100  # int(input('Number of children in each age is '))
n = lam / mu
P_mut = 0.99
mut_level_0 = 1  # float(input('Start mutation level is '))
best = []
tmax = 1000000


def generate(mu, N):
    xi = []
    for j in range(mu):
        xi.append([])
        for i in range(N):
            xi[j].append(random.uniform(xmin, xmax))
    return xi


def fitness(population):
    f = np.zeros(len(population), dtype=float)
    for i in range(len(population)):
        for k in range(N):
            f[i] = f[i] + math.sin(population[i][k]) * math.sin((k+1) * (population[i][k] ** 2) / math.pi) ** 20
        population[i].append(f[i] * (-1))
    population.sort(key=itemgetter(N), reverse=False)
    return population


def selection(lam, mu, X):
    det = np.zeros(len(X), dtype=int)
    det_max = round(2 * n)
    step = det_max / (mu - 1)
    for i in range(len(X)):
        if i > 0:
            det[i] = math.floor(det_max - i * step)
        else:
            det[i] = det_max
    if sum(det) < lam:
        det[mu - 1] = lam - sum(det)
    return det


def cloning(X, DET):
    clones = []
    for i in range(len(DET)):
        k = len(X[i])
        for j in range(DET[i]):
            clones.append(X[i][0:k - 1])
    return clones


def mutation(clones, mut_level):
    for i in range(len(clones)):
        for j in range(len(clones[i])):
            if random.random() < P_mut:
                clones[i][j] = clones[i][j] + random.normalvariate(0, mut_level)
            else:
                clones[i][j] = random.uniform(xmin, xmax)
    return clones


def new_population(X, Y):
    Z = []
    for i in range(len(X)):
        k = len(X[i])
        X[i] = X[i][0:k - 1]
        Z.append(X[i]) 
    for i in range(len(Y)):
        Z.append(Y[i])
    Z = fitness(Z)
    W = []
    for i in range(len(X)):
        k = len(Z[i])
        W.append(Z[i][0:k - 1])
    return W



for i in range(1):
    mut_level = mut_level_0
    print('---------------------------------------------------------------------------------------')
    # generating the firt population
    X = generate(mu, N)
    # print(X)
    # print('--------------------------------------------------------------------------------------')
    for t in range(tmax):
        X = fitness(X)
        # print(X)
        # print('----------------------------------------------------------------------------------')
        DET = selection(lam, mu, X)
        # print(DET)
        # print('----------------------------------------------------------------------------------')
        clon = cloning(X, DET)
        # print(clon)
        # print('----------------------------------------------------------------------------------')
        # mut_level = mut_level_0 / (t+1)
        if t >= 5:
            if best[t - 1][N] == best[t - 6][N]:
                mut_level = min(mut_level * 1.1, mut_level_0)
            if best[t - 1][N] < best[t - 2][N]:
                mut_level = mut_level / 1.5

        Y = mutation(clon, mut_level)
        # print(Y)
        # print('----------------------------------------------------------------------------------')
        X = new_population(X, Y)
        best.append(X[0])
    # print(X)
    # print('---------------------------------------------------------------------------------------')
    q = len(best)
    x = np.arange(0, q - 1, 1)
    y = []
    for i in range(q - 1):
        k = len(best[i])
        y.append(best[i][k - 1])
    print('Best solution is ')
    print(best[q - 1])
    print('Fitness function reached: ')
    print(y[q - 2])
    plt.style.use('classic')
    plt.plot(x, y)
    plt.show()

