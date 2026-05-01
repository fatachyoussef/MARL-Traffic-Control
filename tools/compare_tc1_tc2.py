import pandas as pd
import matplotlib.pyplot as plt

# Chargement des données (Vérifie bien les noms des fichiers)
tc1 = pd.read_csv('results/tc1_high_density/learning_stats.csv') # ou le nom exact de ton CSV TC1
tc2 = pd.read_csv('learning_stats_TC2_HIGH.csv')

plt.figure(figsize=(10, 6))

# Plot TC1 vs TC2
plt.plot(tc1['Step'], tc1['AvgWaitTime'].rolling(window=20).mean(), 
         color='red', label='TC-1 (Local Decision)', linewidth=2)
plt.plot(tc2['Step'], tc2['AvgWaitTime'].rolling(window=20).mean(), 
         color='green', label='TC-2 (Look-ahead Anticipation)', linewidth=2)

# Habillage professionnel
plt.title('Performance : TC-1 vs TC-2 (Haute Densité)', fontsize=14)
plt.xlabel('Cycles de simulation', fontsize=12)
plt.ylabel('Temps d\'attente moyen (Cycles)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.savefig('results/comparaison/comparaison_finale_tc1_tc2.png', dpi=300)
plt.show()