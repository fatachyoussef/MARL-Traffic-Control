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

# --- Zone de lancement ---
if st.sidebar.button("🚀 Lancer la Simulation"):
    st.info(f"Simulation en cours : {agent_choice} sur {topo_choice} ({load_choice})...")
    
    # Placeholder pour les stats en direct
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # --- Appel de ton code main.py ---
    # Note : Pour que cela soit "live" dans Streamlit, 
    # il faudrait normalement ajouter des callbacks, 
    # mais commençons par l'exécution et l'affichage du résultat.
    
    start_time = time.time()
    
    # On lance la fonction que tu as modifiée dans main.py
    # Assure-toi que run_simulation retourne les stats finales ou le nom du CSV
    results = run_simulation(
        topology=topo_choice, 
        agent_type=agent_choice, 
        traffic_load=load_choice, 
        steps=steps_choice
    )
    
    end_time = time.time()
    st.success(f"Simulation terminée en {end_time - start_time:.2f} secondes !")


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

# --- Zone de Comparaison (Bonus) ---
if len(csv_files) >= 2:
    st.divider()
    st.header("⚖️ Comparaison Multi-Modèles")
    selected_models = st.multiselect("Choisir les fichiers à comparer", csv_files, default=csv_files[:2])
    
    if selected_models:
        combined_df = pd.DataFrame()
        for file in selected_models:
            temp_df = pd.read_csv(file)
            temp_df['Model'] = file.replace('learning_stats_', '').replace('.csv', '')
            combined_df = pd.concat([combined_df, temp_df])
        
        # Graphique de comparaison
        st.write("Comparaison de la convergence (Attente vs Temps)")
        # Ici on peut utiliser une librairie comme Plotly pour plus de détails
        st.line_chart(combined_df.pivot(index='Step', columns='Model', values='AvgWaitTime'))