from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn

from .value_iteration import value_iteration

class DeepMaximumEntropy:
    def __init__(self, env, trajectories, features, layers=(8, 16), lr=0.001, discount=0.9):
        self.env = env
        self.trajectories = trajectories
        self.features = torch.from_numpy(features).float()
        self.discount = discount
        self.dynamics = torch.from_numpy(self.env.dynamics).float()

        self.rews = []
        self.net = []
        last = features.shape[1]
        for l in layers:
            self.net.append(nn.Linear(last, l))
            self.net.append(nn.ReLU())
            last = l
        self.net.append(nn.Linear(last, 1))
        self.net.append(nn.Tanh())
        self.net = nn.Sequential(*self.net)
        self.optim = torch.optim.Adagrad(self.net.parameters(), lr)


    def get_rewards(self):
        return self.rews
        

    def forward(self, features):
        x = self.net(features)
        # return (x - x.mean()) / x.std()
        return x

    # old way for computing esv
    # def expected_state_visitation_frequency(self, policy):
    #     # probability of visiting the initial state
    #     prob_initial_state = torch.zeros(self.env.n_states)
    #     for traj in self.trajectories:
    #         prob_initial_state[traj[0, 0]] += 1.0
    #     prob_initial_state = prob_initial_state / self.trajectories.shape[0]
    #     # Compute 𝜇
    #     mu = prob_initial_state.repeat(self.trajectories.shape[1], 1)
    #     for t in range(1, self.trajectories.shape[1]):
    #         mu[t, :] = 0
    #         for s in range(self.env.n_states):
    #             for a in range(self.env.n_actions):
    #                 for s_p in range(self.env.n_states):
    #                     mu[t, s] += mu[t-1, s_p] * policy[s_p, a] * self.dynamics[s_p, a, s]
    #     return mu.sum(dim=0)


    #fast method
    def expected_state_visitation_frequency(self, policy):
        # probability of visiting the initial state
        prob_initial_state = torch.zeros(self.env.n_states)
        for traj in self.trajectories:
            prob_initial_state[traj[0, 0]] += 1.0
        prob_initial_state = prob_initial_state / self.trajectories.shape[0]

        # Compute 𝜇 (pure vectorized :D)
        mu = prob_initial_state.repeat(self.trajectories.shape[1], 1)
        x = (policy[:, :, np.newaxis] * self.dynamics).sum(1)
        for t in range(1, self.trajectories.shape[1]):
            mu[t, :] = mu[t-1, :] @ x

        return mu.sum(dim=0)


    def state_visitation_frequency(self):
        svf = torch.zeros(self.env.n_states, dtype=torch.float32)
        for traj in self.trajectories:
            for s, *_ in traj:
                svf[s] += 1
        return svf / self.trajectories.shape[0]


    def expected_features(self):
        exp_f = torch.zeros_like(self.features[0]).float()
        for traj in self.trajectories:
            for s, *_ in traj:
                exp_f += self.features[s]
        return exp_f / self.trajectories.shape[0]


    def train(self, n_epochs, save_rewards=True, plot=False):            
        self.rews = []
        # exp_f = self.expected_features()
        svf = self.state_visitation_frequency()

        for i in tqdm(range(n_epochs)):
            # with torch.no_grad():
            rewards = self.forward(self.features).flatten()
            if save_rewards:
                self.rews.append(rewards.detach().cpu().numpy())

            policy = value_iteration(0.001, self.env, rewards.detach(), self.discount)
            exp_svf = self.expected_state_visitation_frequency(policy)

            r_grads = svf - exp_svf

            self.optim.zero_grad()
            rewards.backward(-r_grads)
            self.optim.step()

            if plot:
                plt.clf()
                plt.pcolor(rewards.detach().reshape(self.env.grid_size, self.env.grid_size))
                plt.colorbar()
                plt.draw()
                plt.pause(.01)

        plt.show()
        with torch.no_grad():
            rewards = self.forward(self.features)
        return rewards