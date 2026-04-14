import pandas as pd
import matplotlib.pyplot as plt

# 1. Chargement des données
data = pd.read_csv('learning_stats_low.csv')

# 2. Configuration du style
plt.figure(figsize=(10, 6))
plt.plot(data['Step'], data['AvgWaitTime'], color='#2ca02c', linewidth=2, label='Temps d\'attente moyen')

# 3. Ajout d'une moyenne mobile pour lisser la courbe (plus joli pour le rapport)
data['SMA'] = data['AvgWaitTime'].rolling(window=10).mean()
plt.plot(data['Step'], data['SMA'], color='#d62728', linestyle='--', label='Tendance (Moyenne mobile)')

# 4. Habillage du graphique
plt.title('Courbe d\'apprentissage de l\'agent TC-1 (MARL)', fontsize=14)
plt.xlabel('Cycles de simulation (Steps)', fontsize=12)
plt.ylabel('Temps d\'attente moyen (Cycles)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 5. Sauvegarde
plt.savefig('learning_low_curve.png', dpi=300)
print("Graphique sauvegardé sous 'learning_curve.png'")
plt.show()