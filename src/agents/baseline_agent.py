import numpy as np

class FixedTimeAgent:
    def __init__(self, node_id, cycle_time=30):
        """
        Agent Baseline : Change les feux à intervalles réguliers.
        cycle_time : Nombre de cycles (steps) avant de changer d'action.
        """
        self.node_id = node_id
        self.cycle_time = cycle_time
        self.timer = 0
        self.current_action_index = 0

    def select_action(self, intersection, all_agents=None, network=None):
        """
        Ignore l'état du trafic et suit un cycle prédéfini.
        """
        # On incrémente le timer à chaque appel (chaque step de simulation)
        self.timer += 1
        
        # Si on dépasse le temps alloué à l'action actuelle, on passe à la suivante
        if self.timer >= self.cycle_time:
            self.timer = 0
            num_actions = len(intersection.possible_actions)
            # On passe à l'index suivant (0 -> 1 -> 2 ... -> 0)
            self.current_action_index = (self.current_action_index + 1) % num_actions
            
        return self.current_action_index

    # Méthodes vides pour garder la compatibilité avec le main.py
    def update_model(self, state, action, next_state):
        pass

    def save_brain(self, filepath=None):
        pass

    def load_brain(self, filepath=None):
        pass