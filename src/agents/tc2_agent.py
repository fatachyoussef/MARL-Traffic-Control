import numpy as np
import json
class TC2Agent:
    def __init__(self, node_id, gamma=0.99):
        self.node_id = node_id
        self.gamma = gamma
        self.Q_table = {}  # (state, action) -> value
        self.V_table = {}  # state -> value
        
        # --- Variables pour l'apprentissage RTDP ---
        self.counts = {}             # (state, action) -> total_seen
        self.transitions = {}        # (state, action, next_state) -> count
        self.state_seen_counts = {}   # state -> total_seen
        self.state_action_counts = {} # (state, action) -> count

    def get_q_value(self, state, action):
        return self.Q_table.get((state, action), 0.0)

    def get_state_value(self, state):
        return self.V_table.get(state, 0.0)

    # def select_action(self, intersection, all_agents, network):
    #     """ Logique TC-2 : Vote avec anticipation du carrefour suivant """
        
    #     # 1. RÉGLAGE : On crée une liste de la taille RÉELLE du nombre d'actions
    #     num_actions = len(intersection.possible_actions)
    #     gains = [0.0] * num_actions 
        
    #     # 2. RÉGLAGE : On itère sur toutes les actions (0 à 5)
    #     for action in range(num_actions):
    #         impacted_cars = intersection.get_cars_for_action(action)
    #         total_gain = 0.0
            
    #         for car in impacted_cars:
    #             state = car.current_state
    #             q_red = self.get_q_value(state, 'red')
    #             q_green = self.get_q_value(state, 'green')
                
    #             # Look-ahead vers le carrefour suivant
    #             next_stop = network.get_next_stop(self.node_id, car.tl)
    #             v_next = 0.0
    #             if next_stop is not None:
    #                 next_node_id, entry_tl = next_stop
    #                 next_state = (entry_tl, 20, car.destination)
    #                 v_next = all_agents[next_node_id].get_state_value(next_state)
                
    #             # Formule TC-2 : Gain = Q_rouge - (Q_vert + gamma * V_suivant)
    #             car_gain = q_red - (q_green + self.gamma * v_next)
    #             total_gain += car_gain
            
    #         gains[action] = total_gain
            
    #     return np.argmax(gains)
    def select_action(self, intersection, all_agents, network):
        """ Logique TC-2 avec filtrage de panne (Sensor Failure) """
        
        num_actions = len(intersection.possible_actions)
        gains = [0.0] * num_actions 
        
        for action in range(num_actions):
            # 1. On récupère les voitures que l'action pourrait faire passer
            potential_cars = intersection.get_cars_for_action(action)
            
            # 2. FILTRAGE : L'agent ne "voit" que les voitures détectées par les capteurs
            visible_cars = []
            for car in potential_cars:
                # Utilise l'attribut identifié précédemment (car.tl ou car.direction)
                lane_id = car.tl 
                
                # Si get_lane_cars renvoie 0 (cas du Nœud 2), la voiture est invisible
                if intersection.get_lane_cars(lane_id) > 0 or len(intersection.lanes.get(lane_id, [])) == 0:
                    visible_cars.append(car)

            total_gain = 0.0
            
            # 3. Calcul du gain uniquement sur les voitures visibles
            for car in visible_cars:
                state = car.current_state
                q_red = 0 
                q_green = 0
                
                # Look-ahead (Anticipation) : C'est ici que la magie opère !
                # Même si la voiture est sur un carrefour en panne, 
                # on regarde la valeur V du carrefour SUIVANT.
                next_stop = network.get_next_stop(self.node_id, car.tl)
                v_next = 0.0
                if next_stop is not None:
                    next_node_id, entry_tl = next_stop
                    next_state = (entry_tl, 20, car.destination)
                    v_next = all_agents[next_node_id].get_state_value(next_state)
                
                # Formule TC-2
                car_gain = 0 - (0 + self.gamma * v_next)
                total_gain += car_gain
            
            gains[action] = total_gain
            
        return np.argmax(gains)

    def update_model(self, state, action, next_state):
        """ Apprentissage par modèle (RTDP) """
        sa_key = (state, action)
        
        # 1. Mise à jour des fréquences
        self.counts[sa_key] = self.counts.get(sa_key, 0) + 1
        sas_prime_key = (state, action, next_state)
        self.transitions[sas_prime_key] = self.transitions.get(sas_prime_key, 0) + 1

        # 2. Mise à jour pour le calcul de P(L|s)
        self.state_seen_counts[state] = self.state_seen_counts.get(state, 0) + 1
        self.state_action_counts[sa_key] = self.state_action_counts.get(sa_key, 0) + 1

        # 3. Mise à jour de Bellman
        self.perform_rtdp_update(state, action)

    def perform_rtdp_update(self, state, action):
        sa_key = (state, action)
        total_sa = self.counts.get(sa_key, 0)
        if total_sa == 0: return

        q_new = 0.0
        # Somme des (Probabilité * (Récompense + gamma * Valeur_Futur))
        for (s, a, s_prime), count in self.transitions.items():
            if s == state and a == action:
                prob = count / total_sa
                r = 1.0 if s_prime == state else 0.0 # Coût = 1 si la voiture n'a pas bougé
                q_new += prob * (r + self.gamma * self.get_state_value(s_prime))

        self.Q_table[sa_key] = q_new
        self.update_v_value(state)

    def update_v_value(self, state):
        total_obs = self.state_seen_counts.get(state, 0)
        if total_obs == 0: return

        v_new = 0.0
        for act in ['red', 'green']:
            p_L_s = self.state_action_counts.get((state, act), 0) / total_obs
            v_new += p_L_s * self.get_q_value(state, act)
        self.V_table[state] = v_new

    def save_brain(self, filepath="tc2_brain.json"):
        serializable_q = {str(k): v for k, v in self.Q_table.items()}
        serializable_v = {str(k): v for k, v in self.V_table.items()}
        with open(filepath, "w") as f:
            json.dump({"Q": serializable_q, "V": serializable_v}, f)
        print(f"Cerveau TC-2 sauvegardé dans {filepath}")

    def load_brain(self, filepath="tc2_brain.json"):
        with open(filepath, "r") as f:
            data = json.load(f)
            self.Q_table = {eval(k): v for k, v in data["Q"].items()}
            self.V_table = {eval(k): v for k, v in data["V"].items()}
        print("Cerveau TC-2 chargé !")