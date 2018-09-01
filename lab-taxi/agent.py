import numpy as np
from collections import defaultdict
import random

class Agent:

    def __init__(self, env, nA=6, epsilon = 1.0, alpha = 0.05, gamma = 1):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.env = env

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        #return np.random.choice(self.nA)
        rand = random.uniform(0, 1)
        if rand < self.epsilon:
            action = self.env.action_space.sample()
        else:
            try:
                action = np.argmax([self.Q[state][0],self.Q[state][1],self.Q[state][2],self.Q[state][3]])
            except:
                action = self.env.action_space.sample()
        return action


    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        #self.Q[state][action] += 1
        self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])
