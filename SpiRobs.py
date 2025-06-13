import numpy as np
import pandas as pd 
import math 
from scipy.optimize import fsolve 

# Paramètres
# Longueur L de la spirale centrale 
L=100   # mm
taper_angle = 15 
taper_angle_rad=math.radians(taper_angle)
epaisseur_factor = 0.075 # 7.5% (moyenne entre 5% et 10%)

# Résolution de l'équation pour b
def equation(b):
    if b == 0:
        return -taper_angle_rad  # éviter division par 0
    numerator = b * (np.exp(2 * np.pi * b) - 1)
    denominator = np.sqrt(b**2 + 1) * (np.exp(2 * np.pi * b) + 1)
    angle = 2 * np.arctan(numerator / denominator)
    return angle - taper_angle_rad

# Valeur initiale proche (tester avec 0.05, 0.1, etc.)
b_solution = fsolve(equation, x0=0.01 , xtol = 1e-6)[0]
if b_solution <= 0:
    raise ValueError("La solution pour b doit être positive.")
b=b_solution
a=L/(((np.sqrt(b**2+1)/(2*b))*(1+np.exp(2*np.pi*b))*(np.exp(4*np.pi*b)-1)))
print(f"\ntaper angle = {taper_angle} --> b ≈ {b} --> a = {a}")

# Theta, rho, rho_c
thetas=[]
x_or_list=[]
y_or_list=[]
x_ce_list=[]
y_ce_list=[]
epaisseurs = []
largeur_trapeze = []

# Fonction pour rho_ce et s(theta)--> Longueur cumulé
rho_ce_theta = lambda theta: a * (1 + np.exp(2 * np.pi * b)) * np.exp(b * theta) / 2
# s_theta = lambda theta: a*(1+np.exp(2*np.pi*b))/2*np.sqrt(b**2+1)*(np.exp(b*theta)-1)/b

# Discrétisation avec Δθ = 45 
for theta in np.arange(0,4*np.pi*(1+1/16),4*np.pi/16):
    thetas.append(theta)

    # Spirale originale
    rho_or=a*np.exp(b*theta)
    x_or=rho_or*np.cos(theta)
    y_or=rho_or*np.sin(theta)
    x_or_list.append(x_or)
    y_or_list.append(y_or)

    # Spirale centrale 
    rho_ce=rho_ce_theta(theta)
    x_ce=rho_ce*np.cos(theta)
    y_ce=rho_ce*np.sin(theta)
    x_ce_list.append(x_ce)
    y_ce_list.append(y_ce)

    # Les largeurs des trapèzes
    delta = a*np.exp(b*(theta+2*np.pi))-a*np.exp(b*theta)
    delta_adjusted = delta * (1 + epaisseur_factor)
    largeur_trapeze.append(delta_adjusted)
    epaisseur = delta_adjusted * epaisseur_factor
    epaisseurs.append(epaisseur)
    

# Création de DataFrame
df_or=pd.DataFrame({
    "x_or": x_or_list,
    "y_or": y_or_list
})

df_ce=pd.DataFrame({
    "x_ce": x_ce_list,
    "y_ce": y_ce_list
})

df_largeur=pd.DataFrame({
    "theta": thetas,
    "largeur": largeur_trapeze
})

df_epaisseur = pd.DataFrame({
    "e": epaisseurs 
})

# Export au format CSV 
df_or.to_csv("CSV/spirale_originale.csv", index=False)
df_ce.to_csv("CSV/spirale_centrale.csv", index=False)
df_largeur.to_csv("CSV/largeur_trapeze.csv", index=False)
df_epaisseur.to_csv("CSV/epaisseur_trapeze.csv", index = False)
print("Fichier CSV 'spirale_originale.csv' crée")
print("Fichier CSV 'spirale_centrale.csv' crée")
print("Fichier CSV 'largeur_trapeze.csv' crée")
print("Fichier CSV 'epaisseur_trapeze.csv' crée\n")

# Afficher les largeurs des trapèzes en fonction des theta 
print(f"largeur trapeze = {largeur_trapeze} (mm) \n")


# Recalculer L 
a1=a*(1+np.exp(2*np.pi*b))/2
L_test=a1*np.sqrt(b**2+1)*(np.exp(b*4*np.pi)-1)/b 
print(f"Pour b = {b} , a = {a} --> L = {L_test}\n")

# Coordonnées des trapèzes
trapeze_coords = []
for i in range(len(thetas)-1):
    theta1, theta2 = thetas[i], thetas[i+1]
    rho1_ce, rho2_ce = rho_ce_theta(theta1), rho_ce_theta(theta2)
    largeur1, largeur2 = largeur_trapeze[i], largeur_trapeze[i+1]
    x1, y1 = rho1_ce*np.cos(theta1), rho1_ce * np.sin(theta1)
    x2, y2 = rho2_ce * np.cos(theta2), rho2_ce * np.sin(theta2)
    trapeze_coords.append([(x1-largeur1/2,y1), (x1+largeur1/2,y1),
                            (x2+largeur2/2,y2), (x2+largeur2/2,y2)])
df_trapeze = pd.DataFrame(trapeze_coords, columns=["p1","p2","p3","p4"])
df_trapeze.to_csv("CSV/trapeze_coords.csv", index=False)
print("Fichier CSV 'trapeze_coords.csv' crée\n")




