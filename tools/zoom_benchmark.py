import pandas as pd
import matplotlib.pyplot as plt

# Chargement uniquement des agents IA
tc1 = pd.read_csv('learning_stats_TC1_HIGH_v2.csv')
tc2 = pd.read_csv('learning_stats_TC2_HIGH_v2.csv')

plt.figure(figsize=(10, 6))

# On utilise une fenêtre de lissage plus petite pour voir les détails
window = 100

plt.plot(tc1['Step'], tc1['AvgWaitTime'].rolling(window=window).mean(), 
         color='red', label='TC-1: Local Decision', linewidth=1.5)

plt.plot(tc2['Step'], tc2['AvgWaitTime'].rolling(window=window).mean(), 
         color='green', label='TC-2: Look-ahead Anticipation', linewidth=1.5)

# --- FOCUS SUR LA ZONE UTILE ---
# On force l'axe Y à rester entre 3.8 et 4.5 pour voir les micro-différences
plt.ylim(3.8, 4.5) 

plt.title('Comparaison Haute Précision : TC-1 vs TC-2', fontsize=14)
plt.xlabel('Cycles de simulation', fontsize=12)
plt.ylabel('Temps d\'attente moyen (Cycles)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.savefig('zoom_ia_comparison.png', dpi=300)
plt.show()