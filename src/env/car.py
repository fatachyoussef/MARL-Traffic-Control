class Car:
    def __init__(self, car_id, start_tl, destination):
        self.id = car_id
        self.tl = start_tl
        self.place = 20  # Capacité maximale de la file [cite: 56]
        self.destination = destination
        
        # Pour le calcul du temps de trajet total (statistiques finales)
        self.waiting_time = 0 
        
        # Stockage de l'état précédent pour l'apprentissage de l'agent
        self.last_state = None

    @property
    def current_state(self):
        """ Retourne l'état s = (tl, p, d) utilisé par l'agent. """
        return (self.tl, self.place, self.destination)

    def update_position(self, next_place, next_tl):
        """ 
        Met à jour la position et sauvegarde l'ancien état.
        Indispensable pour perform_rtdp_update(state, action, reward).
        """
        self.last_state = self.current_state
        self.place = next_place
        self.tl = next_tl

    def __repr__(self):
        return f"Car({self.id}, pos={self.place}, tl={self.tl}, dest={self.destination})"