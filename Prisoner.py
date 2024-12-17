# Prisoner RL Agent

import numpy as np
# Q_table
#              ACTIONS
# OBSERV [_, _]
# ATIONS [_, _]

# Q_table 
# Actions: 0 = Stay Silent, 1 = Snitch
# States: Snitched on count, Round number
class q_table_prisoner:
    def __init__(self, n_actions=2, n_states= 2, alpha=0.1, gamma=0.6, epsilon=0.1):
        # Probability distribution of actions for each state
        self.q_table = np.zeros((n_actions, n_states))

        # Hyperparameters
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount factor
        self.epsilon = epsilon # Exploration rate

    # Choose an action based on the epsilon-greedy policy
    def choose_action(self, state):
        self.state = state
        # Exploration
        if np.random.uniform(0, 1) < self.epsilon:
            self.action = np.random.choice(len(self.q_table))
        
        # Exploitation
        else:
            self.action = np.argmax(self.q_table[:, state])
        
        return self.action
        
    # Update the Q-table
    def update(self, reward, next_state):
        # Bellman equation
        self.q_table[self.action, self.state] = self.q_table[self.action, self.state] + self.alpha * (reward + self.gamma * np.max(self.q_table[:, next_state]) - self.q_table[self.action, self.state])
        # Reduce the exploration rate
        self.epsilon = max(0.05, self.epsilon * 0.999)