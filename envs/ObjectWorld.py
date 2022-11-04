import math
from itertools import product

import torch
import numpy as np

from DeepMaximumEntropy.value_iteration import value_iteration


class WorldObject(object):
    def __init__(self, inner_color, outer_color):
        self.inner_color = inner_color
        self.outer_color = outer_color
        

class Objectworld:
    def __init__(self, grid_size, n_objects, n_colors, wind, discount):
        self.wind = float(wind)
        self.grid_size = grid_size
        self.actions = ((1, 0), (0, 1), (-1, 0), (0, -1), (0, 0))
        self.n_actions = len(self.actions)
        self.n_states = grid_size**2
        self.n_objects = n_objects
        self.n_colors = n_colors
        self.discount = discount

        self.objects = {}
        for _ in range(self.n_objects):
            obj = WorldObject(np.random.randint(self.n_colors), 
                              np.random.randint(self.n_colors))
            while True:
                x = np.random.randint(self.grid_size)
                y = np.random.randint(self.grid_size)

                if (x, y) not in self.objects:
                    break
            self.objects[x, y] = obj

        self.dynamics = self.transition_probabilities()
        self.real_rewards = np.array([self.reward(s) for s in range(self.n_states)])
            

    def feature_vector(self, state, discrete=True):
        x_s, y_s = state%self.grid_size, state//self.grid_size

        nearest_inner = {}
        nearest_outer = {}

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if (x, y) in self.objects:
                    dist = math.hypot((x - x_s), (y - y_s))
                    obj = self.objects[x, y]
                    if obj.inner_color in nearest_inner:
                        if dist < nearest_inner[obj.inner_color]:
                            nearest_inner[obj.inner_color] = dist
                    else:
                        nearest_inner[obj.inner_color] = dist
                    if obj.outer_color in nearest_outer:
                        if dist < nearest_outer[obj.outer_color]:
                            nearest_outer[obj.outer_color] = dist
                    else:
                        nearest_outer[obj.outer_color] = dist

        for c in range(self.n_colors):
            if c not in nearest_inner:
                nearest_inner[c] = 0
            if c not in nearest_outer:
                nearest_outer[c] = 0

        if discrete:
            state = np.zeros((2*self.n_colors*self.grid_size,))
            i = 0
            for c in range(self.n_colors):
                for d in range(1, self.grid_size+1):
                    if nearest_inner[c] < d:
                        state[i] = 1
                    i += 1
                    if nearest_outer[c] < d:
                        state[i] = 1
                    i += 1
        else:
            state = np.zeros((2*self.n_colors))
            i = 0
            for c in range(self.n_colors):
                state[i] = nearest_inner[c]
                i += 1
                state[i] = nearest_outer[c]
                i += 1

        return state


    def feature_matrix(self, discrete=True):
        return np.array([self.feature_vector(i, discrete)
                         for i in range(self.n_states)])


    def reward(self, state_p):
        x, y = state_p%self.grid_size, state_p//self.grid_size

        near_c0 = False
        near_c1 = False
        for (dx, dy) in product(range(-3, 4), range(-3, 4)):
            if 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size:
                if (abs(dx) + abs(dy) <= 3 and
                        (x+dx, y+dy) in self.objects and
                        self.objects[x+dx, y+dy].outer_color == 0):
                    near_c0 = True
                if (abs(dx) + abs(dy) <= 2 and
                        (x+dx, y+dy) in self.objects and
                        self.objects[x+dx, y+dy].outer_color == 1):
                    near_c1 = True
        if near_c0 and near_c1:
            return 1
        if near_c0:
            return -1
        return 0
        

    def reset(self, random=False):
        if random:
            self.state = np.random.randint(self.n_states)
        else:
            self.state = 0
        return self.state


    def step(self, a):
        probs = self.dynamics[:, a, self.state]
        self.state = np.random.choice(self.n_states, p=probs)
        return self.state


    def transition_probabilities(self):
        dynamics = np.zeros((self.n_states, self.n_actions, self.n_states))
        # S_t+1, A_t, S_t
        for s in range(self.n_states):
            x, y = s%self.grid_size, s//self.grid_size
            for a in range(self.n_actions):
                x_a, y_a = self.actions[a]
                for d in range(self.n_actions):
                    x_d, y_d = self.actions[d]
                    if 0 <= x+x_d < self.grid_size and 0 <= y+y_d < self.grid_size:
                        dynamics[(x+x_d) + (y+y_d)*self.grid_size, a, s] += self.wind/self.n_actions
                    else:
                        dynamics[s, a, s] += self.wind/self.n_actions
                if 0 <= x+x_a < self.grid_size and 0 <= y+y_a < self.grid_size:
                    dynamics[(x+x_a) + (y+y_a)*self.grid_size, a, s] += 1 - self.wind
                else:
                    dynamics[s, a, s] += 1 - self.wind
                
        return dynamics


    def optimal_policy(self):
        real_rewards = torch.tensor([self.reward(s) for s in range(self.n_states)], dtype=torch.float32)
        policy = value_iteration(0.0001, self, real_rewards, self.discount)
        return policy.argmax(1)


    def generate_trajectories(self, num, length, policy=None):
        if not policy:
            policy = self.optimal_policy()

        trajs = []
        for n in range(num):
            t = []
            state = self.reset(random=True)
            for i in range(length):
                action = policy[state]
                state_p = self.step(action)
                t.append([state, action, self.reward(state_p)])
                state = state_p
            trajs.append(t)
        return np.array(trajs)