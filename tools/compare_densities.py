import pandas as pd
import matplotlib.pyplot as plt

# Chargement des deux fichiers
high = pd.read_csv('learning_stats.csv')
low = pd.read_csv('learning_stats_low.csv')

plt.figure(figsize=(10, 6))

# Plot Densité Haute
plt.plot(high['Step'], high['AvgWaitTime'].rolling(window=10).mean(), 
         color='red', label='Densité Haute (1-8 voitures)', linewidth=2)

# Plot Densité Basse
plt.plot(low['Step'], low['AvgWaitTime'].rolling(window=10).mean(), 
         color='blue', label='Densité Basse (1-4 voitures)', linewidth=2)

# Habillage
plt.title('Influence de la densité du trafic sur l\'apprentissage TC-1', fontsize=14)
plt.xlabel('Cycles de simulation', fontsize=12)
plt.ylabel('Temps d\'attente moyen (Cycles)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.savefig('comparaison_densite.png', dpi=300)
print("Graphe de comparaison sauvegardé !")
plt.show()