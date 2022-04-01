from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def __init__(self, observation_space, action_space, step_size, discount, epsilon=0.1, seed=None):
        """Instantiates a new agent

        Args:
            observation_space (OpenAI gym observation space): the observation space of the environment to run
            action_space (OpenAI gym action space): the action space of the environment to run
            step_size (float): float in the range [0, 1] that specifies the size of each update
            discount (float): gamma. How much to discount future rewards
            epsilon (float, optional): percentage of exploratory actions. Defaults to 0.1.
            values (np.ndarray, optional): Array specifying the values for each action at each state. Defaults to np.zeros(state_n, action_n).
            seed (int, optional): Seed to use for the random number generator. Defaults to None.
        """

    @abstractmethod
    def reset_train(self):
        """Resets the agent's learning procedure, leaving it like an empty canvas.
        """

    @abstractmethod
    def reset(self):
        """Resets the state of the agent to start a new episode
        """
        
    @abstractmethod
    def act(self, state: int, reward: int, done=False) -> int:
        """Chooses an action according to previous observation and reward. Additionally, updates its values following the Expected SARSA formulation.

        Arguments:
            state (int): environment state
            reward (int): reward obtained from previous action
        Returns:
            int: Action to take
        """
        