import numpy as np
import json

class TC1Agent:
    def __init__(self, node_id, gamma=0.99):
        self.node_id = node_id  # <--- Ajoute cette ligne
        self.gamma = gamma
        self.V_table = {}  
        self.Q_table = {}  
        self.transitions = {} 
        self.counts = {} 
        
        # Initialisation propre des compteurs de probabilité du feu [cite: 117]
        self.state_seen_counts = {}
        self.state_action_counts = {}

    def get_state_value(self, state):
        return self.V_table.get(state, 0.0)

    def get_q_value(self, state, action):
        return self.Q_table.get((state, action), 0.0)

    def compute_gain(self, car_state):
        # Gain = Q(s, red) - Q(s, green) 
        return self.get_q_value(car_state, 'red') - self.get_q_value(car_state, 'green')
    
    def select_action(self, intersection):
        # On ne récupère plus bêtement toutes les voitures
        # On demande à l'intersection quelles voitures sont visibles par les capteurs
        all_cars = intersection.get_all_cars()
        
        # FILTRAGE POUR LA PANNE :
        # L'agent ne prend en compte que les voitures sur les voies où le capteur fonctionne
        visible_cars = []
        for car in all_cars:
            lane_id = getattr(car, 'direction', None) or getattr(car, 'lane_id', None)
            if lane_id is None:
                lane_id = car.tl
            # On vérifie si la voie de la voiture est fonctionnelle au carrefour actuel
            if intersection.get_lane_cars(lane_id) > 0 or len(intersection.lanes.get(lane_id, [])) == 0:
                visible_cars.append(car)
            # Si get_lane_cars renvoie 0 alors qu'il y a des voitures, 
            # la voiture est ignorée dans le calcul du gain.
        
        possible_actions = intersection.possible_actions
        best_action_index = 0 
        max_total_gain = -float('inf')

        for i, action in enumerate(possible_actions):
            total_gain = 0
            # On calcule le gain uniquement sur les voitures "visibles"
            for car in visible_cars:
                if car.tl in action: 
                    total_gain += self.compute_gain(car.current_state)
            
            if total_gain > max_total_gain:
                max_total_gain = total_gain
                best_action_index = i 
                
        return best_action_index

    # def select_action(self, intersection):
    #     intersection_cars = intersection.get_all_cars()
    #     possible_actions = intersection.possible_actions
        
    #     best_action_index = 0 # On va stocker l'index maintenant
    #     max_total_gain = -float('inf')

    #     # On itère sur les index (0, 1, 2...)
    #     for i, action in enumerate(possible_actions):
    #         total_gain = 0
    #         for car in intersection_cars:
    #             if car.tl in action: 
    #                 total_gain += self.compute_gain(car.current_state)
            
    #         if total_gain > max_total_gain:
    #             max_total_gain = total_gain
    #             best_action_index = i # On stocke l'index i
                
    #     return best_action_index # On renvoie l'entier i, pas la liste [0,1]

    def update_model(self, state, action, next_state):
        """ Regroupe toute la logique d'apprentissage [cite: 127, 156] """
        # 1. Mise à jour des probabilités de transition P(s'|s,a) [cite: 158]
        sa_key = (state, action)
        self.counts[sa_key] = self.counts.get(sa_key, 0) + 1
        
        sas_prime_key = (state, action, next_state)
        self.transitions[sas_prime_key] = self.transitions.get(sas_prime_key, 0) + 1

        # 2. Mise à jour de la probabilité P(L|s) [cite: 117]
        self.state_seen_counts[state] = self.state_seen_counts.get(state, 0) + 1
        self.state_action_counts[sa_key] = self.state_action_counts.get(sa_key, 0) + 1

        # 3. Calcul de Bellman (Q) et mise à jour de V [cite: 121, 126]
        self.perform_rtdp_update(state, action)

    def update_v_value(self, state):
        total_obs = self.state_seen_counts.get(state, 0)
        if total_obs == 0: return

        v_new = 0.0
        for act in ['red', 'green']:
            p_L_s = self.state_action_counts.get((state, act), 0) / total_obs
            v_new += p_L_s * self.get_q_value(state, act)
        self.V_table[state] = v_new

    def perform_rtdp_update(self, state, action):
        sa_key = (state, action)
        total_sa = self.counts.get(sa_key, 0)
        if total_sa == 0: return

        q_new = 0.0
        # On itère sur les transitions connues pour cet état-action [cite: 121]
        for key, count in self.transitions.items():
            if key[0] == state and key[1] == action:
                s_prime = key[2]
                prob = count / total_sa
                # R = 1 si immobile, 0 si mouvement [cite: 119]
                r = 1.0 if s_prime == state else 0.0
                q_new += prob * (r + self.gamma * self.get_state_value(s_prime))

        self.Q_table[sa_key] = q_new
        self.update_v_value(state)


    def save_tables(self, filename="tables.json"):
        # On convertit les clés (tuples) en chaînes de caractères pour le JSON
        data = {
            "Q_table": {str(k): v for k, v in self.Q_table.items()},
            "V_table": {str(k): v for k, v in self.V_table.items()}
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"Tables sauvegardées dans {filename}")

    def load_tables(self, filename="tables.json"):
        with open(filename, 'r') as f:
            data = json.load(f)
            # On peut reconstruire les tuples ici si nécessaire, 
            # ou adapter la recherche pour lire les strings.
            self.Q_table = {eval(k): v for k, v in data["Q_table"].items()}
            self.V_table = {eval(k): v for k, v in data["V_table"].items()}

    

    def save_brain(self, filepath="tc1_brain.json"):
        # Convertir les clés (tuples) en strings pour JSON
        serializable_q = {str(k): v for k, v in self.Q_table.items()}
        serializable_v = {str(k): v for k, v in self.V_table.items()}
        
        with open(filepath, "w") as f:
            json.dump({"Q": serializable_q, "V": serializable_v}, f)
        print(f"Cerveau sauvegardé dans {filepath}")

    def load_brain(self, filepath="tc1_brain.json"):
        with open(filepath, "r") as f:
            data = json.load(f)
            # Reconvertir les strings en tuples
            self.Q_table = {eval(k): v for k, v in data["Q"].items()}
            self.V_table = {eval(k): v for k, v in data["V"].items()}
        print("Cerveau chargé avec succès !")