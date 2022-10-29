import numpy as np

def value_iteration(threshold, env, rewards, discount=0.01):
    transition = env.dynamics
    V = np.zeros(env.n_states)
    delta = np.inf

    while delta > threshold:
        delta = 0
        for s in range(env.n_states):
            max_v = -np.inf
            for a in range(env.n_actions):
                probs = np.zeros((env.n_states))
                for s_p in range(env.n_states):
                    probs[s_p] = transition(s_p, a, s)
                max_v = max(max_v, np.dot(probs, rewards + discount * V))
            delta = max(delta, abs(V[s] - max_v))
            V[s] = max_v

    policy = np.zeros((env.n_states, env.n_actions))
    for s in range(env.n_states):
        for a in range(env.n_actions):
            probs = np.zeros((env.n_states))
            for s_p in range(env.n_states):
                probs[s_p] = transition(s_p, a, s)
            policy[s, a] = np.dot(probs, rewards + discount * V)

    policy = policy - policy.max(axis=1, keepdims=True)
    exps = np.exp(policy)
    policy = exps / exps.sum(axis=1, keepdims=True)
    return policy