import numpy as np

from .agent import Agent
from .utils import argmax

class ExpectedDynaQPlus(Agent):
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

        policy = np.array([self.epsilon] * action_n) / action_n
        policy[max_action] += 1 - self.epsilon

        if self.prev_state is not None:
            self.update_model(self.prev_state, self.prev_action, state, reward, done)
            self._update_counts(self.prev_state, self.prev_action)
            self.plan(num_steps=self.plan_steps)
            
            if done:
                exp_p = 0
            else:
                exp_p = np.dot(values, policy)
            update = reward + self.discount * exp_p - self.values[self.prev_state, self.prev_action]
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
        out_dist = {k: v/total_samples for k, v in out_samples.items()}
        choices = list(out_dist.keys())
        p = list(out_dist.values())
        state, reward, done = tuple(self.rand_gen.choice(choices, p=p))
        return int(state), reward, done

    def _update_counts(self, state, action):
        update = np.ones(self.tau.shape)
        self.tau += update
        self.tau[state, action] = 0

    def plan(self, num_steps=10):
        for _ in range(num_steps):
            prev_state, prev_action = self.rand_gen.choice(list(self.model.keys()))
            state, reward, done = self.sample_model(prev_state, prev_action)

            values = self.values[state]
            action_n = self.action_space.n
            max_action = argmax(values)
            policy = np.array([self.epsilon] * action_n) / action_n
            policy[max_action] += 1 - self.epsilon

            if self.prev_state is not None:
                self.update_model(self.prev_state, self.prev_action, state, reward, done)
                self._update_counts(self.prev_state, self.prev_action)
            
            if done:
                exp_p = 0
            else:
                exp_p = np.dot(values, policy)
            update = reward + self.discount * exp_p - self.values[self.prev_state, self.prev_action]
            self.values[self.prev_state, self.prev_action] += self.step_size * update



