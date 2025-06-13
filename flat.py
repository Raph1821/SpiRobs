import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.integrate import simpson 

# Charger les points de la spirale centrale
df = pd.read_csv("spirale_centrale.csv")
x = df['x_ce'].values
y = df['y_ce'].values

# Calculer les longueurs cumulées le long de la spirale
dx = np.diff(x)
dy = np.diff(y)
ds = np.sqrt(dx**2 + dy**2)
s = np.concatenate([[0], np.cumsum(ds)])  # Longueur cumulée

# Facteur de correction pour compenser la perte géométrique de 2,55 %
correction_factor = 100 / 97.4534697198329  # 100 mm / 97.45 mm

# Aplatir la spirale
x_flat = np.zeros_like(s)  # Ligne droite verticale
y_flat = s*correction_factor 

# Sauvegarder l’aplatissement
df_flat = pd.DataFrame({'x': x_flat, 'y': y_flat})
df_flat.to_csv("spiral_flat.csv", index=False)

# Affichage pour vérification
plt.figure(figsize=(12, 4))
plt.plot(x_flat, y_flat, 'b-')
plt.title("Spirale aplatie (déroulée)")
plt.xlabel("Longueur cumulée (mm)")
plt.ylabel("Y aplati")
plt.grid(True)
plt.show()



