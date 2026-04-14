class Intersection:
    def __init__(self, node_id):
        self.node_id = node_id
        # 8 feux par carrefour : 4 (tout droit/droite) + 4 (gauche) [cite: 45]
        # On utilise un dictionnaire pour stocker les voitures par identifiant de feu (tl)
        self.lanes = {tl_id: [] for tl_id in range(8)} 
        
        # Définition des 6 actions possibles (combinaisons de feux au vert) [cite: 47, 48]
        self.possible_actions = self._define_actions()

    def _define_actions(self):
        """ 
        Définit les combinaisons de feux qui peuvent être verts simultanément. 
        Empêche les collisions entre les flux de trafic[cite: 46, 47].
        """
        return [
            [0, 1], [2, 3], # Nord/Sud tout droit OU Est/Ouest tout droit
            [4, 5], [6, 7], # Tourner à gauche (N/S) OU Tourner à gauche (E/O)
            [0, 4], [2, 6]  # Combinaisons spécifiques selon le modèle du papier [cite: 47]
        ]

    def get_all_cars(self):
        """ Récupère toutes les instances de Car présentes à ce carrefour. """
        all_cars = []
        for lane_cars in self.lanes.values():
            all_cars.extend(lane_cars)
        return all_cars

    def add_car_to_lane(self, car, tl_id):
        """ 
        Ajoute une voiture à la fin de la file (place 20) si possible.
        Chaque file a une capacité limitée à 20 voitures[cite: 56, 174].
        """
        if len(self.lanes[tl_id]) < 20:
            self.lanes[tl_id].append(car)
            return True
        return False # Carrefour saturé (voiture refusée) [cite: 174]

    def move_cars_internal(self, active_actions, network):
        """
        Gère le mouvement discret des voitures.
        active_actions: Liste des IDs des feux actuellement au VERT pour ce nœud.
        network: L'instance TrafficNetwork pour gérer les transitions entre nœuds.
        """
        rewards_to_learn = [] # Liste de tuples pour l'apprentissage RTDP

        for tl_id, lane in self.lanes.items():
            # AMÉLIORATION : Trier les voitures de la place 1 à 20 
            # pour traiter la tête de file en premier (évite l'effet de chevauchement)
            lane.sort(key=lambda x: x.place)
            
            # On utilise une copie de la liste pour pouvoir faire des .pop() sans erreur
            for car in list(lane):
                old_state = car.current_state
                action_taken = 'green' if tl_id in active_actions else 'red'
                
                # CAS 1 : La voiture est en tête (place 1) et le feu est VERT [cite: 77]
                if car.place == 1 and action_taken == 'green':
                    next_stop = network.get_next_stop(self.node_id, tl_id)
                    
                    if next_stop is None: # La voiture quitte la ville (escape) [cite: 78]
                        lane.remove(car)
                        # Reward = 0 (succès), next_state = None (fin de l'épisode)
                        rewards_to_learn.append((car, old_state, 'green', 0, None))
                    else:
                        target_node, target_tl = next_stop
                        target_intersection = network.intersections[target_node]
                        
                        # Tentative d'entrée dans le carrefour suivant [cite: 57]
                        if target_intersection.add_car_to_lane(car, target_tl):
                            lane.remove(car)
                            car.update_position(20, target_tl)
                            # Reward = 0 (succès), progression vers un nouvel état [cite: 119]
                            rewards_to_learn.append((car, old_state, 'green', 0, car.current_state))
                        else: 
                            # File suivante pleine : la voiture reste bloquée à la place 1 [cite: 57]
                            rewards_to_learn.append((car, old_state, 'green', 1, old_state))

                # CAS 2 : La voiture est derrière et tente d'avancer [cite: 54]
                elif car.place > 1:
                    # Vérifier si la place immédiatement devant (place - 1) est libre
                    if not any(other.place == car.place - 1 for other in lane):
                        car.update_position(car.place - 1, tl_id)
                        # Reward = 0 car la voiture a progressé [cite: 119, 157]
                        rewards_to_learn.append((car, old_state, action_taken, 0, car.current_state))
                    else: 
                        # Bloquée par la voiture de devant : reward = 1 (coût d'attente) [cite: 119]
                        rewards_to_learn.append((car, old_state, action_taken, 1, old_state))
                
                # CAS 3 : Bloquée à la place 1 car le feu est ROUGE [cite: 55]
                else:
                    rewards_to_learn.append((car, old_state, 'red', 1, old_state))
        
        return rewards_to_learn