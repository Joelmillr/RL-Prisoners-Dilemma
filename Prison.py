# Prisoners Dilemma in Gymnasium

import gymnasium as gym

# Creating a gym environment for the Prisoners Dilemma
# The environment has two agents, A and B, that can take actions 0 or 1
# The rewards are as follows:
# - Both agents stay silent: -1 for both
# - Agent A snitches, Agent B stays silent: -3 for A, 0 for B
# - Agent A stays silent, Agent B snitches: 0 for A, -3 for B
# - Both agents snitch: -2 for both
# The observations are the number of times the other agent has snitched, and the round number

class PrisonersDilemma(gym.Env):
    # Initialize the environment
    def __init__(self, n_rounds):
        self.n_rounds = n_rounds
        self.round = 0
        self.agent_actions = [0, 0] # For rendering
        self.observation_space_a = 0
        self.observation_space_b = 0

    # Reset the environment
    def reset(self):
        self.round = 0
        self.agent_actions = [0,0]
        self.observation_space_a = 0
        self.observation_space_b = 0

        return (self.observation_space_a, self.observation_space_b)

    # Step function
    def step(self, action_a, action_b):
        reward_a = 0
        reward_b = 0

        # Calculate the rewards based on the actions of the agents
        if action_a == 0 and action_b == 0:
            reward_a = -1
            reward_b = -1
        elif action_a == 0 and action_b == 1:
            reward_a = -3
            reward_b = 0
        elif action_a == 1 and action_b == 0:
            reward_a = 0
            reward_b = -3
        elif action_a == 1 and action_b == 1:
            reward_a = -2
            reward_b = -2

        # Store actions for rendering
        self.agent_actions = [action_a, action_b]

        # Store the actions of the other agent and the round number in the observation space
        self.observation_space_a = action_b
        self.observation_space_b = action_a

        # Increment the round number
        self.round += 1

        return (
            self.round,
            self.observation_space_a, 
            self.observation_space_b,
            reward_a, 
            reward_b,
            self.round >= self.n_rounds,
        )

    def render(self):
        # Print the round number and the actions of the agents
        print("Round:", self.round)
        print(f"Agent A {'snitched' if self.agent_actions[0] else 'stayed silent'}")
        print(f"Agent B {'snitched' if self.agent_actions[1] else 'stayed silent'}")