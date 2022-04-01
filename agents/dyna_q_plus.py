import numpy as np

from .agent import Agent
from .utils import argmax

class DynaQPlus(Agent):
    def __init__(self, observation_space, action_space, step_size, discount, epsilon=0.1, kappa=0.001, plan_steps=10, seed=None):
        self.observation_space = observation_space
        self.action_space = action_space
        self.step_size = step_size
        self.discount = discount
        self.epsilon = epsilon
        self.kappa = kappa
        self.plan_steps = plan_steps
        self.prev_action = None
        self.prev_state = None
        self.rand_gen = np.random.default_rng(seed)

        self.reset_train()

    def reset_train(self):
        n_states = self.observation_space.n
        n_actions = self.action_space.n
        self.values = np.zeros((n_states, n_actions))
        self.tau = np.zeros((n_states, n_actions))
        self.model = {}

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

        if self.prev_state is not None:
            self.update_model(self.prev_state, self.prev_action, state, reward, done)
            self._update_counts(self.prev_state, self.prev_action)
            self.plan(num_steps=self.plan_steps)

            if done:
                max_p = 0
            else:
                max_p = values.max()
            update = reward + self.discount * max_p - self.values[self.prev_state, self.prev_action]
            self.values[self.prev_state, self.prev_action] += self.step_size * update

        self.prev_action = action
        self.prev_state = state

        return action
        
    def update_model(self, prev_state, prev_action, state, reward, done):
        in_tuple = (prev_state, prev_action)
        out_tuple = (state, reward, done)

        if in_tuple not in self.model:
            self.model[in_tuple] = {}
        
        self.model[in_tuple][out_tuple] = self.model[in_tuple].get(out_tuple, 0) + 1

    def sample_model(self, state, action):
        in_tuple = (state, action)

        out_samples = self.model[in_tuple]
        total_samples = sum(out_samples.values())
        choices = list(out_samples.keys())
        p = np.array(list(out_samples.values())) / total_samples
        state, reward, done = tuple(self.rand_gen.choice(choices, p=p))
        return int(state), reward, done

    def _update_counts(self, state, action):
        update = np.ones(self.tau.shape)
        self.tau += update
        self.tau[state, action] = 0

    def plan(self, num_steps=10):
        for _ in range(num_steps):
            prev_state, prev_action = self.rand_gen.choice(list(self.model.keys()))
            # print("PREV STATE, PREV ACTION")
            # print(prev_state, prev_action)
            state, reward, done = self.sample_model(prev_state, prev_action)
            # print("STATE, REWARD, DONE")
            # print(state, reward, done)

            values = self.values[state]

            if done:
                q_max = 0
            else:
                q_max = np.max(values)

            # print("Q-MAX")
            # print(q_max)

            reward += self.kappa * np.sqrt(self.tau[prev_state, prev_action])
            update = reward + self.discount * q_max - self.values[prev_state, prev_action]
            # print("BEFORE")
            # print(self.values)
            self.values[prev_state, prev_action] += self.step_size * update
            # print("AFTER")
            # print(self.values)


