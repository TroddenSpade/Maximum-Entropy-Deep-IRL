import numpy as np
import matplotlib.pyplot as plt

from MaximumEntropy.MaximumEntropy import MaximumEntropy
from envs.GridWorld import GridWorld


if __name__ == "__main__" :
    grid_size = 5
    random_rate = 0.2
    num_trajectories = 20
    len_trajectory = 15

    n_epochs = 200
    discount = 0.01
    lr = 0.01

    gw = GridWorld(grid_size, random_rate)
    trajs = gw.generate_trajectories(num_trajectories, len_trajectory)
    features = np.eye(gw.n_states)
    me = MaximumEntropy(gw, trajs, features, lr, discount)

    rewards = me.train(n_epochs).reshape(grid_size, grid_size)
    true_rewards = np.array([gw.reward(s) for s in range(grid_size*grid_size)]).reshape(grid_size, grid_size)

    plt.subplot(1, 2, 1)
    plt.pcolor(true_rewards)
    plt.colorbar()
    plt.title("Real reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(rewards)
    plt.colorbar()
    plt.title("Generated reward")
    plt.show()