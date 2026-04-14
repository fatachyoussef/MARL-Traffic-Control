import random
from tqdm import tqdm
from env.network import TrafficNetwork
from env.car import Car
from agents.tc1_agent import TC1Agent
import csv


def run_simulation(steps=50000):
    # 1. Initialisation
    network = TrafficNetwork()
    agent = TC1Agent(gamma=0.99) # [cite: 185]
    car_id_counter = 0
    total_waiting_time = 0
    cars_exited = 0
    refused_cars = 0
    history = []

    # Points d'entrée de la ville (bords de la grille 2x3)
    # Feux qui n'ont pas de destination interne dans ton dictionnaire connections
    entry_points = [(0, 0), (0, 3), (2, 0), (2, 2), (3, 1), (3, 3), (5, 1), (5, 2)]

    print(f"Lancement de la simulation TC-1 pour {steps} cycles...")

    for step in tqdm(range(steps)):
        # --- PHASE A : Génération de trafic ---
        # Insertion de 1 à 8 voitures par cycle [cite: 173]
        # num_new_cars = random.randint(1, 8) # ici pour la 1ere partie du rapport, on peut réduire la charge pour mieux visualiser l'apprentissage de l'agent
        num_new_cars = random.randint(1, 4)  # On divise par 2 la charge maximale
        for _ in range(num_new_cars):
            start_node_id, start_tl = random.choice(entry_points)
            destination = random.randint(1, 10) # 10 destinations possibles [cite: 81]
            
            new_car = Car(car_id_counter, start_tl, destination)
            intersection = network.intersections[start_node_id]
            
            # Tentative d'ajout dans la file d'attente
            if intersection.add_car_to_lane(new_car, start_tl):
                car_id_counter += 1
            else:
                refused_cars += 1 # Carrefour saturé [cite: 174]

        # --- PHASE B : Prise de décision (Le Vote) ---
        node_decisions = {}
        for node_id, intersection in network.intersections.items():
            all_cars_in_node = intersection.get_all_cars()
            # L'agent choisit l'action qui maximise le gain cumulé [cite: 101, 103]
            chosen_action = agent.select_action(all_cars_in_node, intersection.possible_actions)
            node_decisions[node_id] = chosen_action

        # --- PHASE C : Mouvement et Apprentissage ---
        for node_id, intersection in network.intersections.items():
            active_lights = node_decisions[node_id]
            
            # move_cars_internal gère la physique et retourne les données d'apprentissage
            results = intersection.move_cars_internal(active_lights, network)
            
            for car, old_state, action, reward, next_state in results:
                # Apprentissage Model-Based (RTDP) [cite: 127]
                # Si next_state est None, la voiture est sortie de la ville
                if next_state is None:
                    cars_exited += 1
                    # Pour une sortie, le futur attendu est 0
                    agent.update_model(old_state, action, next_state=(None, None, car.destination))
                else:
                    agent.update_model(old_state, action, next_state)
                
                # Statistiques
                if reward == 1:
                    total_waiting_time += 1
        # Dans la boucle (tous les 500 steps par exemple) :
        if step % 500 == 0 and cars_exited > 0:
            current_avg = total_waiting_time / cars_exited
            history.append((step, current_avg))

        # À la fin, sauvegarde en CSV
        
        
    # --- PHASE D : Rapport final ---
    print("\n--- RÉSULTATS DE LA SIMULATION ---")
    print(f"Voitures ayant quitté la ville : {cars_exited}")
    print(f"Voitures refusées (saturation) : {refused_cars}")
    
    # with open("learning_stats.csv", "w", newline="") as f:
    with open("learning_stats_low.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "AvgWaitTime"])
            writer.writerows(history)
    if cars_exited > 0:
        avg_wait = total_waiting_time / cars_exited
        print(f"Temps d'attente moyen : {avg_wait:.2f} cycles")

if __name__ == "__main__":
    run_simulation()