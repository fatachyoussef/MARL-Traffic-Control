import pandas as pd
import matplotlib.pyplot as plt

# 1. Chargement du fichier généré par ton dernier run
data = pd.read_csv('learning_stats_TC1_HIGH_v2.csv')

# 2. Configuration
plt.figure(figsize=(10, 6))
plt.plot(data['Step'], data['AvgWaitTime'], color='#2ca02c', alpha=0.3, label='Brut')

# Moyenne mobile pour voir la tendance de convergence
data['SMA'] = data['AvgWaitTime'].rolling(window=20).mean()
plt.plot(data['Step'], data['SMA'], color='#1f77b4', linewidth=2, label='Tendance TC-2')

# 3. Habillage
plt.title('Courbe d\'apprentissage TC-1 (Look-ahead) - Haute Densité', fontsize=14)
plt.xlabel('Cycles de simulation', fontsize=12)
plt.ylabel('Temps d\'attente moyen (Cycles)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 4. Sauvegarde
plt.savefig('results/tc2_high_density/learning_curve_tc1_v2.png', dpi=300)
print("Graphique TC-1 sauvegardé !")
plt.show()