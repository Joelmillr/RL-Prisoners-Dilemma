# Prisoner RL Agent
# Deep Q-learning Agent that takes the round number and the number of times the other agent has snitched as input

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Decision class that takes the number of rounds and the number of times the other agent has snitched as input
# and returns the decision to snitch or not
class Decision(nn.Module):
    # Initialize the Decision class
    def __init__(self):
        super(Decision, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    # Forward pass of the Decision class
    def forward(self, n_rounds, n_snitched):
        # convert the inputs to a tensor
        n_rounds = torch.tensor([n_rounds], dtype=torch.float32)
        n_snitched = torch.tensor([n_snitched], dtype=torch.float32)

        # concatenate the inputs
        x = torch.cat((n_rounds, n_snitched), dim=0)
        # pass the inputs through the network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Prisoner class that takes the number of rounds as input
# and returns the decision to snitch or not
class Prisoner:
    # Initialize the Prisoner class
    def __init__(self, n_rounds):
        self.n_rounds = n_rounds
        self.decision = Decision()
        self.optimizer = optim.Adam(self.decision.parameters(), lr=0.001)
        self.gamma = 0.99
        self.eps = 0.5

    def reset(self):
        self.eps = 0.5

    # Act method that takes the number of rounds and the number of times the other agent has snitched as input
    # and returns the decision to snitch or not
    def act(self, observation):
        self.n_rounds = observation[0]
        self.n_snitched = observation[1]
        # Epsilon-greedy policy
        if torch.rand(1).item() < self.eps:
            return torch.randint(0, 2, (1,))
        else:
            return torch.argmax(self.decision(self.n_rounds, self.n_rounds))

    # Train method that takes the number of rounds, the number of times the other agent has snitched, the action,
    # the reward, the number of rounds_, and the number of times the other agent has snitched_ as input
    # and returns the loss
    def train(self, observation_, reward):
        n_rounds_ = observation_[0]
        n_snitched_ = observation_[1]

        # Compute the target
        target = reward + self.gamma * torch.max(self.decision(n_rounds_, n_snitched_))
        target = target.unsqueeze(0)

        # Compute the loss
        loss = F.mse_loss(self.decision(self.n_rounds, self.n_rounds), target)
        # Optimize the Decision class
        self.optimizer.zero_grad()
        # Backward pass to compute the gradients
        loss.backward()
        # Update the weights
        self.optimizer.step()

        # Decay the epsilon
        self.eps *= 0.99
        
        return loss.item()
