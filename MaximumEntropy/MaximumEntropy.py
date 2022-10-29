import numpy as np
from tqdm import tqdm

from envs.GridWorld import GridWorld
from functions.value_iteration import value_iteration


class MaximumEntropy:
    def __init__(self, env, trajectories, features, lr=0.01, discount=0.01):
        self.env = env
        self.features = features
        self.trajectories = trajectories
        self.num_t = trajectories.shape[0]
        self.len_t = trajectories.shape[1]

        self.theta = np.random.uniform(size=(env.n_states,))
        self.discount = discount
        self.lr = lr


    def expected_features(self):
        exp_f = np.zeros_like(self.features[0])
        for traj in self.trajectories:
            for s, *_ in traj:
                exp_f += self.features[s]
        return exp_f / self.num_t


    def expected_state_visitation_frequency(self, policy):
        # probability of visiting the initial state
        prob_initial_state = np.zeros(self.env.n_states)
        for traj in self.trajectories:
            prob_initial_state[traj[0, 0]] += 1.0
        prob_initial_state = prob_initial_state / self.num_t

        # Compute 𝜇
        mu = np.repeat([prob_initial_state], self.len_t, axis=0)
        for t in range(1, self.len_t):
            mu[t, :] = 0
            for s in range(self.env.n_states):
                for a in range(self.env.n_actions):
                    for s_p in range(self.env.n_states):
                        mu[t, s] += mu[t-1, s_p] * policy[s_p, a] * self.env.dynamics(s_p, a, s)
        return mu.sum(axis=0)


    def train(self, n_epochs):
        exp_f = self.expected_features()

        for i in tqdm(range(n_epochs)):
            rewards = self.features.dot(self.theta)
            policy = value_iteration(self.discount, self.env, rewards)
            exp_svf = self.expected_state_visitation_frequency(policy)

            grads = exp_f - exp_svf @ self.features
            self.theta += self.lr * grads

        return self.features.dot(self.theta)