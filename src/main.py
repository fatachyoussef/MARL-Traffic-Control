import random
import csv
from tqdm import tqdm
from env.network import TrafficNetwork
from env.car import Car
from agents.tc1_agent import TC1Agent
from agents.tc2_agent import TC2Agent
from agents.baseline_agent import FixedTimeAgent
# ==========================================
# CONFIGURATION DE L'EXPÉRIENCE
# ==========================================
AGENT_TYPE = "TC2"    # Options: "TC1", "TC2", "FIXED"
TRAFFIC_LOAD = "HIGH"  # Options: "HIGH" (1-8 cars) ou "LOW" (1-4 cars)
STEPS = 200000
# ==========================================

def run_simulation():
    # 1. Initialisation du réseau et des agents
    network = TrafficNetwork()
    
    # Création d'un dictionnaire d'agents (un par intersection)
    agents = {}
    for i in range(6):
        if AGENT_TYPE == "TC1":
            agents[i] = TC1Agent(node_id=i)
        elif AGENT_TYPE == "TC2":
            agents[i] = TC2Agent(node_id=i)
        else:
            # On définit un cycle de 30 steps par exemple
            agents[i] = FixedTimeAgent(node_id=i, cycle_time=30)
    
    car_id_counter = 0
    total_waiting_time = 0
    cars_exited = 0
    refused_cars = 0
    history = []

    print(f"Lancement simulation {AGENT_TYPE} | Densité: {TRAFFIC_LOAD} | Cycles: {STEPS}")

    for step in tqdm(range(STEPS)):
        # --- PHASE A : Génération de trafic ---
        max_cars = 8 if TRAFFIC_LOAD == "HIGH" else 4
        num_new_cars = random.randint(1, max_cars)
        
        for _ in range(num_new_cars):
            # On utilise les points d'entrée définis dans network.py
            start_node_id, start_tl = random.choice(network.entry_points)
            destination = random.randint(1, 10)
            
            new_car = Car(car_id_counter, start_tl, destination)
            intersection = network.intersections[start_node_id]
            
            if intersection.add_car_to_lane(new_car, start_tl):
                car_id_counter += 1
            else:
                refused_cars += 1

        # --- PHASE B : Prise de décision (Le Vote) ---
        node_decisions = {}
        for node_id, intersection in network.intersections.items():
            if AGENT_TYPE == "TC1":
                # TC1 utilise uniquement les données locales
                node_decisions[node_id] = agents[node_id].select_action(intersection)
            else:
                # TC2 utilise les agents voisins et le réseau pour l'anticipation
                node_decisions[node_id] = agents[node_id].select_action(intersection, agents, network)

        # --- PHASE C : Mouvement et Apprentissage ---
        for node_id, intersection in network.intersections.items():
            action_index = node_decisions[node_id]
            # active_lights = node_decisions[node_id]
            # results = intersection.move_cars_internal(active_lights, network)
            active_lights = intersection.possible_actions[action_index]
            results = intersection.move_cars_internal(active_lights, network)
            for car, old_state, action, reward, next_state in results:
                # Chaque agent apprend de ses propres voitures
                if next_state is None:
                    cars_exited += 1
                    agents[node_id].update_model(old_state, action, next_state=(None, None, car.destination))
                else:
                    agents[node_id].update_model(old_state, action, next_state)
                
                if reward == 1:
                    total_waiting_time += 1

        # Collecte des stats tous les 500 cycles
        if step % 500 == 0 and cars_exited > 0:
            current_avg = total_waiting_time / cars_exited
            history.append((step, current_avg))

    # --- PHASE D : Sauvegarde et Rapport ---
    # Nom de fichier automatique selon la config
    csv_filename = f"learning_stats_{AGENT_TYPE}_{TRAFFIC_LOAD}_v2.csv"
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "AvgWaitTime"])
        writer.writerows(history)
    
    # Sauvegarde des "cerveaux"
    for i in range(6):
        agents[i].save_brain(f"models/tc1_v2/{AGENT_TYPE}_node_{i}.json")

    print(f"\n--- RÉSULTATS {AGENT_TYPE} ({TRAFFIC_LOAD}) ---")
    print(f"Voitures sorties : {cars_exited}")
    print(f"Refusées : {refused_cars}")
    if cars_exited > 0:
        print(f"Attente moyenne : {total_waiting_time / cars_exited:.2f} cycles")

if __name__ == "__main__":
    run_simulation()