import numpy as np
import torch


def value_iteration(threshold, env, rewards, discount=0.01):
    V = torch.zeros(env.n_states, dtype=torch.float32)
    delta = np.inf
    
    while delta > threshold:
        delta = 0
        for s in range(env.n_states):
            max_v = torch.tensor([-float('inf')])
            for a in range(env.n_actions):
                probs = torch.from_numpy(env.dynamics[:, a, s]).float()
                max_v = torch.maximum(max_v, torch.dot(probs, rewards + discount * V))
            delta = max(delta, torch.abs(V[s] - max_v).numpy())
            V[s] = max_v

    policy = torch.zeros((env.n_states, env.n_actions), dtype=torch.float32)
    for s in range(env.n_states):
        for a in range(env.n_actions):
            probs = torch.from_numpy(env.dynamics[:, a, s]).float()
            policy[s, a] = torch.dot(probs, rewards + discount * V)

    policy = policy - policy.max(dim=1, keepdims=True)[0]
    exps = torch.exp(policy)
    policy = exps / exps.sum(dim=1, keepdims=True)
    return policy