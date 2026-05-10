import streamlit as st
import pandas as pd
import time
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
# Importe ta fonction de simulation modifiée
from main import run_simulation 
from env.network import TrafficNetwork # Pour accéder à la logique du réseau

# --- Configuration de la page ---
st.set_page_config(page_title="MARL Traffic Control Dashboard", layout="wide")

def draw_topology(topology_type):
    # 1. On crée une instance temporaire pour lire les connexions
    temp_net = TrafficNetwork(topology_type=topology_type)
    connections = temp_net.connections
    
    # 2. Création du graphe dirigé (les flèches indiquent le sens de circulation)
    G = nx.DiGraph()
    
    # 3. Ajout des arêtes à partir du dictionnaire connections
    # connections est (node_src, tl_src) -> (node_dest, tl_dest)
    for (node_src, tl_src), (node_dest, tl_dest) in connections.items():
        G.add_edge(node_src, node_dest)
    
    # 4. Définition des positions des nœuds
    # Pour la GRID, on peut forcer une disposition rectangulaire
    if topology_type == "GRID":
        pos = {0: (0, 1), 1: (1, 1), 2: (2, 1),
               3: (0, 0), 4: (1, 0), 5: (2, 0)}
    else:
        # Pour COMPLEX, on laisse NetworkX calculer une forme aérée
        pos = nx.spring_layout(G, seed=42) 
    
    # 5. Création de la figure Matplotlib
    fig, ax = plt.subplots(figsize=(8, 5))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', 
            node_size=2000, edge_color='gray', arrows=True, 
            arrowsize=20, font_size=15, font_weight='bold', ax=ax, connectionstyle='arc3,rad=0.1')
    
    plt.title(f"Visualisation de la Topologie : {topology_type}")
    return fig

st.title("🚦 MARL Traffic Control - Dashboard de Simulation")
st.markdown("""
Cette plateforme permet de tester la résilience et l'efficacité des agents **TC1 (Local)** et **TC2 (Anticipation)** sur différentes topologies urbaines.
""")

# --- Barre latérale : Configuration ---
st.sidebar.header("⚙️ Configuration de l'Expérience")

topo_choice = st.sidebar.selectbox(
    "Choisir la Topologie",
    ("GRID", "COMPLEX")
)

agent_choice = st.sidebar.selectbox(
    "Type d'Agent",
    ("TC1", "TC2", "FIXED")
)

load_choice = st.sidebar.select_slider(
    "Densité du Trafic",
    options=["LOW", "HIGH"]
)

steps_choice = st.sidebar.number_input(
    "Nombre de Cycles (Steps)",
    min_value=1000,
    max_value=500000,
    value=10000,
    step=1000
)

# # --- Zone de lancement ---
# if st.sidebar.button("🚀 Lancer la Simulation"):
#     st.info(f"Simulation en cours : {agent_choice} sur {topo_choice} ({load_choice})...")
    
#     # Placeholder pour les stats en direct
#     progress_bar = st.progress(0)
#     status_text = st.empty()
    
#     # --- Appel de ton code main.py ---
#     # Note : Pour que cela soit "live" dans Streamlit, 
#     # il faudrait normalement ajouter des callbacks, 
#     # mais commençons par l'exécution et l'affichage du résultat.
    
#     start_time = time.time()
    
#     # On lance la fonction que tu as modifiée dans main.py
#     # Assure-toi que run_simulation retourne les stats finales ou le nom du CSV
#     results = run_simulation(
#         topology=topo_choice, 
#         agent_type=agent_choice, 
#         traffic_load=load_choice, 
#         steps=steps_choice
#     )
    
#     end_time = time.time()
#     st.success(f"Simulation terminée en {end_time - start_time:.2f} secondes !")

# --- Zone de lancement ---
if st.sidebar.button("🚀 Lancer la Simulation"):
    st.info(f"Simulation en cours : {agent_choice} sur {topo_choice}...")
    
    # Placeholders
    progress_bar = st.sidebar.progress(0)
    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()

    # Utiliser une liste pour stocker les données (les listes sont mutables en Python, 
    # donc on peut les modifier sans 'nonlocal')
    history_steps = []
    history_waits = []

    def update_live_dashboard(current_step, current_wait, total_steps):
        # On ajoute les données aux listes
        history_steps.append(current_step)
        history_waits.append(current_wait)
        
        # Mise à jour de l'interface
        progress = current_step / total_steps
        progress_bar.progress(progress)
        
        # Création d'un DataFrame temporaire pour l'affichage
        current_df = pd.DataFrame({
            "Step": history_steps,
            "AvgWaitTime": history_waits
        }).set_index("Step")
        
        with chart_placeholder.container():
            st.line_chart(current_df)
        
        with metrics_placeholder.container():
            st.metric("Attente moyenne actuelle", f"{current_wait:.2f} cycles")

    # Appel de la simulation
    run_simulation(
        topology=topo_choice, 
        agent_type=agent_choice, 
        traffic_load=load_choice, 
        steps=steps_choice,
        st_callback=update_live_dashboard 
    )
    
    st.success("Simulation terminée !")

    # --- NOUVEAU : PANNEAU DE MÉTRIQUES FINALES ---
    st.divider()
    st.subheader("🏁 Synthèse des Performances Finales")
    
    final_wait = history_waits[-1] if history_waits else 0
    
    # On crée 3 colonnes pour les métriques
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.metric("Temps d'attente Final", f"{final_wait:.2f} cycles")
    
    with m2:
        # Calcul de la stabilité (différence entre le début et la fin)
        initial_wait = history_waits[0] if history_waits else 0
        improvement = initial_wait - final_wait
        st.metric("Gain d'Apprentissage", f"{improvement:.2f} cycles", delta=f"{improvement:.2f}")

    with m3:
        # Estimation du débit
        st.metric("Statut du Modèle", "Sauvegardé", help=f"Modèle JSON stocké dans le dossier models/{agent_choice.lower()}_v2/")

    # Ajout d'un conseil d'expert basé sur le résultat
    if final_wait > 8:
        st.warning("⚠️ Le réseau semble saturé. L'anticipation (TC2) pourrait aider à réguler les flux entrants.")
    elif final_wait < 4:
        st.info("✅ Excellente fluidité. Le modèle a convergé vers une politique optimale.")


st.subheader(f"Plan du réseau : {topo_choice}")
fig_map = draw_topology(topo_choice)
st.pyplot(fig_map)

# --- Affichage des Résultats ---
st.divider()
st.header("📊 Analyse des Performances")



# On cherche les fichiers CSV générés dans le dossier
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

if csv_files:
    selected_csv = st.selectbox("Sélectionner un fichier de stats pour analyse", csv_files)
    
    if selected_csv:
        df = pd.read_csv(selected_csv)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Évolution de l'Attente Moyenne")
            st.line_chart(df.set_index('Step')['AvgWaitTime'])
            
        with col2:
            st.subheader("Statistiques Brutes")
            st.write(df.tail(10)) # Affiche les derniers cycles
            
            # Petit résumé métrique
            last_wait = df['AvgWaitTime'].iloc[-1]
            st.metric(label="Attente Finale Moyenne", value=f"{last_wait:.2f} cycles")
else:
    st.warning("Aucun fichier de données (.csv) trouvé. Lancez une simulation pour générer des résultats.")

# # --- Zone de Comparaison (Bonus) ---
# if len(csv_files) >= 2:
#     st.divider()
#     st.header("⚖️ Comparaison Multi-Modèles")
#     selected_models = st.multiselect("Choisir les fichiers à comparer", csv_files, default=csv_files[:2])
    
#     if selected_models:
#         combined_df = pd.DataFrame()
#         for file in selected_models:
#             temp_df = pd.read_csv(file)
#             temp_df['Model'] = file.replace('learning_stats_', '').replace('.csv', '')
#             combined_df = pd.concat([combined_df, temp_df])
        
#         # Graphique de comparaison
#         st.write("Comparaison de la convergence (Attente vs Temps)")
#         # Ici on peut utiliser une librairie comme Plotly pour plus de détails
#         st.line_chart(combined_df.pivot(index='Step', columns='Model', values='AvgWaitTime'))

# --- Zone de Comparaison (Expert) ---
if len(csv_files) >= 2:
    st.divider()
    st.header("⚖️ Comparaison Multi-Modèles & ROI")
    
    # 1. Sélection précise pour le Duel
    col_a, col_b = st.columns(2)
    with col_a:
        model_a = st.selectbox("Modèle de Référence (Base)", csv_files, key="duel_a")
    with col_b:
        model_b = st.selectbox("Modèle Challenger (Test)", csv_files, key="duel_b")

    if model_a and model_b:
        df_a = pd.read_csv(model_a)
        df_b = pd.read_csv(model_b)

        # --- CALCUL DU ROI (Return on Investment / Gain de performance) ---
        # On prend la moyenne des 10% dernières étapes pour comparer la stabilité finale
        last_n = max(1, int(len(df_a) * 0.1))
        avg_a = df_a['AvgWaitTime'].tail(last_n).mean()
        avg_b = df_b['AvgWaitTime'].tail(last_n).mean()
        
        # Calcul du pourcentage d'amélioration
        # Un chiffre positif signifie que B est meilleur (temps plus bas)
        roi_improvement = ((avg_a - avg_b) / avg_a) * 100

        # Affichage du ROI avec un grand indicateur
        st.subheader("🎯 Analyse du Gain de Performance (ROI)")
        if roi_improvement > 0:
            st.success(f"L'agent challenger réduit le temps d'attente de **{roi_improvement:.2f}%** par rapport au modèle de référence.")
        else:
            st.warning(f"L'agent challenger est **{abs(roi_improvement):.2f}%** moins efficace que le modèle de référence.")

        # --- TABLEAU DE DUEL (Head-to-Head) ---
        st.subheader("⚔️ Tableau de Duel")
        
        duel_data = {
            "Métrique": [
                "Temps d'attente final (moyen)", 
                "Temps d'attente maximum", 
                "Stabilité (Écart-type)",
                "Écart par rapport à la référence"
            ],
            "Modèle A (Base)": [
                f"{avg_a:.2f} cycles",
                f"{df_a['AvgWaitTime'].max():.2f} cycles",
                f"{df_a['AvgWaitTime'].std():.3f}",
                "-"
            ],
            "Modèle B (Challenger)": [
                f"{avg_b:.2f} cycles",
                f"{df_b['AvgWaitTime'].max():.2f} cycles",
                f"{df_b['AvgWaitTime'].std():.3f}",
                f"{roi_improvement:+.2f}%"
            ]
        }
        st.table(pd.DataFrame(duel_data))

        # --- GRAPHIQUE COMPARATIF SUPERPOSÉ ---
        st.subheader("📈 Comparaison des Courbes de Convergence")
        
        # On synchronise les étapes pour le graphique
        comparison_df = pd.DataFrame({
            "Step": df_a["Step"],
            "Modèle A (Base)": df_a["AvgWaitTime"],
            "Modèle B (Challenger)": df_b["AvgWaitTime"]
        }).set_index("Step")
        
        st.line_chart(comparison_df)