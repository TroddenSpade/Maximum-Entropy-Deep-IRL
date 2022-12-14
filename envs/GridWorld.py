import numpy as np

class GridWorld:
    def __init__(self, grid_size=5, wind=0.2):
        # (0,0) bottom left corner (x,y)
        self.names = ["Right", "Down", "Left", "Up"]
        self.actions = [(1,0),(0,1),(-1,0),(0,-1)]
        self.n_actions = len(self.actions)
        self.n_states = grid_size**2
        self.wind = float(wind)
        
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size))

        self.features = np.eye(self.n_states)
        self.dynamics = self.transition_probabilities()
        self.real_rewards = np.array([self.reward(s) for s in range(self.n_states)])
        self.state = 0


    def reward(self, state_p):
        return 1 if state_p == self.n_states-1 else 0
        

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

    
    def test(self):
        for s in range(self.n_states):
            print("/// State: ", s)
            for a in range(self.n_actions):
                print("/// Action: ", self.names[a])
                probs = self.dynamics[:, a, s]
                print(probs.reshape(-1, self.grid_size))


    def optimal_policy(self, state):
        x, y = state%self.grid_size, state//self.grid_size
        if x > y:
            return 1
        elif x < y:
            return 0
        else:
            return np.random.randint(2)


    def generate_trajectories(self, num, length, policy=None):
        if not policy:
            policy = self.optimal_policy

        trajs = []
        for n in range(num):
            t = []
            state = self.reset()
            for i in range(length):
                action = policy(state)
                state_p = self.step(action)
                t.append([state, action])
                state = state_p
            trajs.append(t)
        return np.array(trajs)