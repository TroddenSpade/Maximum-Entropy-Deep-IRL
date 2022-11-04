import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from envs.GridWorld import GridWorld
from functions.value_iteration import value_iteration


class MaximumEntropy:
    def __init__(self, env, trajectories, features, lr=0.002, discount=0.9):
        self.env = env
        self.features = features
        self.trajectories = trajectories
        self.rews = []

        self.theta = np.random.uniform(size=(features.shape[1],))
        self.discount = discount
        self.lr = lr


    def get_rewards(self):
        return self.rews


    def expected_features(self):
        exp_f = np.zeros_like(self.features[0])
        for traj in self.trajectories:
            for s, *_ in traj:
                exp_f += self.features[s]
        return exp_f / self.trajectories.shape[0]


    def expected_state_visitation_frequency(self, policy):
        # probability of visiting the initial state
        prob_initial_state = np.zeros(self.env.n_states)
        for traj in self.trajectories:
            prob_initial_state[traj[0, 0]] += 1.0
        prob_initial_state = prob_initial_state / self.trajectories.shape[0]

        # Compute 𝜇
        mu = np.repeat([prob_initial_state], self.trajectories.shape[1], axis=0)
        for t in range(1, self.trajectories.shape[1]):
            mu[t, :] = 0
            for s in range(self.env.n_states):
                for a in range(self.env.n_actions):
                    for s_p in range(self.env.n_states):
                        mu[t, s] += mu[t-1, s_p] * policy[s_p, a] * self.env.dynamics[s_p, a, s]
        return mu.sum(axis=0)


    def train(self, n_epochs, save_rewards=True, plot=False):
        self.rews = []
        exp_f = self.expected_features()

        for i in tqdm(range(n_epochs)):
            rewards = self.features.dot(self.theta)
            if save_rewards:
                self.rews.append(rewards)
                
            policy = value_iteration(self.discount, self.env, rewards)
            exp_svf = self.expected_state_visitation_frequency(policy)

            grads = exp_f - exp_svf @ self.features
            self.theta += self.lr * grads

            if plot:
                plt.clf()
                plt.pcolor(rewards.reshape(self.env.grid_size, self.env.grid_size))
                plt.colorbar()
                plt.draw()
                plt.pause(.01)

        plt.show()

        return self.features.dot(self.theta)