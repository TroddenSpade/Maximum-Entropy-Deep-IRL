import matplotlib.pyplot as plt

from DeepMaximumEntropy.DeepMaximumEntropy import DeepMaximumEntropy
from envs.ObjectWorld import Objectworld


grid_size = 10
n_objects = 10
n_colors = 2
wind = 0.3
discount = 0.9

num_trajectories, len_trajectory = 500, 30
n_epochs = 500
lr = 0.005

ow = Objectworld(grid_size, n_objects, n_colors, wind, discount)
trajectories = ow.generate_trajectories(num_trajectories, len_trajectory)
features = ow.feature_matrix(discrete=False)

dme = DeepMaximumEntropy(ow, trajectories, features, (16, 16), lr, discount)

plt.pcolor(ow.real_rewards.reshape(grid_size, grid_size))
plt.colorbar()
plt.title("Real reward")
plt.show()

rewards = dme.train(n_epochs, plot=True).reshape(grid_size, grid_size)

plt.pcolor(rewards)
plt.colorbar()
plt.title("Real reward")
plt.show()