import pandas as pd
import matplotlib.pyplot as plt

# Chargement des trois expériences
# Note : assure-toi que les noms de fichiers correspondent à tes CSV
try:
    fixed = pd.read_csv('learning_stats_FIXED_HIGH.csv')
    tc1 = pd.read_csv('learning_stats_TC1_HIGH_v2.csv')
    tc2 = pd.read_csv('learning_stats_TC2_HIGH_v2.csv')
except FileNotFoundError as e:
    print(f"Erreur : Un des fichiers est manquant. {e}")

plt.figure(figsize=(12, 7))

# Plot des trois courbes avec moyenne mobile pour la lisibilité
window = 50
plt.plot(fixed['Step'], fixed['AvgWaitTime'].rolling(window=window).mean(), 
         color='grey', label='Baseline: Fixed-Time (Standard)', linestyle='--', linewidth=2)

plt.plot(tc1['Step'], tc1['AvgWaitTime'].rolling(window=window).mean(), 
         color='red', label='TC-1: Local MARL', linewidth=2)

plt.plot(tc2['Step'], tc2['AvgWaitTime'].rolling(window=window).mean(), 
         color='green', label='TC-2: Multi-Agent Anticipation', linewidth=3)

# Habillage
plt.yscale('log') # Optionnel : utilise une échelle log si TC-2 est trop "écrasé" en bas
plt.title('Benchmark Final : IA vs Méthodes Traditionnelles', fontsize=16)
plt.xlabel('Cycles de simulation', fontsize=12)
plt.ylabel('Temps d\'attente moyen (Cycles) - Échelle Log', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend(fontsize=11)

plt.savefig('benchmark_final_master.png', dpi=300)
plt.show()