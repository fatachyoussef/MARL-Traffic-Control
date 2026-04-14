from env.intersection import Intersection

class TrafficNetwork:
    def __init__(self):
        # Création des 6 intersections (0 à 5) disposées en 2x3
        self.intersections = {i: Intersection(i) for i in range(6)}
        
        # Table de routage : (node_actuel, feu_pris) -> (prochain_node, feu_entree)
        self.connections = self._setup_connections()
        
        # Liste des points d'entrée (bords de la ville)
        # Ces feux ne reçoivent pas de trafic interne et servent à l'injection
        self.entry_points = [
            (0, 0), (0, 3), # Node 0 : Nord, Ouest
            (1, 0),         # Node 1 : Nord
            (2, 0), (2, 2), # Node 2 : Nord, Est
            (3, 1), (3, 3), # Node 3 : Sud, Ouest
            (4, 1),         # Node 4 : Sud
            (5, 1), (5, 2)  # Node 5 : Sud, Est
        ]

    def _setup_connections(self):
        """
        Définit la topologie Manhattan 2x3.
        Ligne 1 : 0 --- 1 --- 2
        Ligne 2 : 3 --- 4 --- 5
        Convention TL : 0=Nord, 1=Sud, 2=Est, 3=Ouest
        """
        connections = {
            # --- LIGNE 1 (Horizontale) ---
            (0, 2): (1, 3), # Node 0 Est -> Node 1 Ouest
            (1, 3): (0, 2), # Node 1 Ouest -> Node 0 Est
            (1, 2): (2, 3), # Node 1 Est -> Node 2 Ouest
            (2, 3): (1, 2), # Node 2 Ouest -> Node 1 Est

            # --- LIGNE 2 (Horizontale) ---
            (3, 2): (4, 3), # Node 3 Est -> Node 4 Ouest
            (4, 3): (3, 2), # Node 4 Ouest -> Node 3 Est
            (4, 2): (5, 3), # Node 4 Est -> Node 5 Ouest
            (5, 3): (4, 2), # Node 5 Ouest -> Node 4 Est

            # --- CONNEXIONS VERTICALES ---
            (0, 1): (3, 0), # Node 0 Sud -> Node 3 Nord
            (3, 0): (0, 1), # Node 3 Nord -> Node 0 Sud
            (1, 1): (4, 0), # Node 1 Sud -> Node 4 Nord
            (4, 0): (1, 1), # Node 4 Nord -> Node 1 Sud
            (2, 1): (5, 0), # Node 2 Sud -> Node 5 Nord
            (5, 0): (2, 1), # Node 5 Nord -> Node 2 Sud
        }
        return connections

    def get_next_stop(self, current_node_id, current_tl):
        """ 
        Retourne (next_node_id, entry_tl) ou None si la voiture 
        quitte le réseau ("escapes the city").
        """
        return self.connections.get((current_node_id, current_tl))