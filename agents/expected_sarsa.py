import numpy as np

from .agent import Agent
from .utils import argmax

class ExpectedSarsa(Agent):
    def __init__(self, observation_space, action_space, step_size, discount, epsilon=0.1, seed=None):
        self.observation_space = observation_space
        self.action_space = action_space
        self.step_size = step_size
        self.discount = discount
        self.epsilon = epsilon
        self.prev_action = None
        self.prev_state = None
        self.rand_gen = np.random.default_rng(seed)

        self.reset_train()

    def reset_train(self):
        n_states = self.observation_space.n
        n_actions = self.action_space.n
        self.values = np.zeros((n_states, n_actions))

    def reset(self):
        """Resets the state of the agent.
        """
        
        self.prev_action = None
        self.prev_state = None
        

    def act(self, state: int, reward: int, done=False) -> int:
        """Chooses an action according to previous observation and reward. Additionally, updates its values following the Expected SARSA formulation.

        Arguments:
            state (int): environment state
            reward (int): reward obtained from previous action
        Returns:
            int: Action to take
        """
        values = self.values[state]
        action_n = self.action_space.n
        max_action = argmax(values)
        if self.rand_gen.random() < self.epsilon:
            action = self.rand_gen.integers(action_n)
        else:
            action = max_action

        policy = np.array([self.epsilon] * action_n) / action_n
        policy[max_action] += 1 - self.epsilon

        if self.prev_state is not None:
            if done:
                exp_p = 0
            else:
                exp_p = np.dot(values, policy)
            update = reward + self.discount * exp_p - self.values[self.prev_state, self.prev_action]
            self.values[self.prev_state, self.prev_action] += self.step_size * update

        self.prev_action = action
        self.prev_state = state
        return action
        