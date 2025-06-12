import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Paramètres des spirales
a_values = [4, 6, 8]
b_values = [0.2, 0.5, 0.8]

fig, axes = plt.subplots(len(b_values), len(a_values), figsize=(12, 10), subplot_kw={'aspect': 'equal'})

theta = np.linspace(0, 2 * np.pi, 500)

for i, b in enumerate(b_values):  # lignes = b
    max_rho_line = max([np.max(a * np.exp(b * theta)) for a in a_values])
    for j, a in enumerate(a_values):  # colonnes = a
        rho = a * np.exp(b * theta)
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)

        ax = axes[i, j]
        ax.plot(x, y, label=f"a={a}, b={b}")

        # Afficher axes x=0 et y=0
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)

        # Afficher les valeurs x et y (ticks)
        ax.set_xlim(-max_rho_line, max_rho_line)
        ax.set_ylim(-max_rho_line, max_rho_line)
        ax.set_title(f"a={a}, b={b}", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)

        # Rendre les ticks plus lisibles
        ax.tick_params(axis='both', labelsize=8)

# Ajouter flèches directionnelles
fig.text(0.5, 0.98, "a augmente", ha='center', va='center', fontsize=14, weight='bold')
fig.text(0.095, 0.5, "b\naugmente", ha='center', va='center', fontsize=14, weight='bold', rotation=90)

arrow_a = FancyArrowPatch((0.2, 0.96), (0.8, 0.96), transform=fig.transFigure,
                          arrowstyle='->', mutation_scale=20, lw=2)
arrow_b = FancyArrowPatch((0.12, 0.8), (0.12, 0.2), transform=fig.transFigure,
                          arrowstyle='->', mutation_scale=20, lw=2)
fig.add_artist(arrow_a)
fig.add_artist(arrow_b)

plt.tight_layout(rect=[0.08, 0.05, 0.95, 0.95])
plt.show()
