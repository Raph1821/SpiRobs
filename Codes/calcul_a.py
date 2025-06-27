import numpy as np
L = 100

b = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

def calcul_a(b):
    num = L * b
    racine = np.sqrt(1 + b**2)
    fact = np.exp(b * 3*np.pi) - np.exp(b * np.pi)
    den = racine * fact
    return num/den

a = calcul_a(b)
print(a)

print(calcul_a(0.2230))