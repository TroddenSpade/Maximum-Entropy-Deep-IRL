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
    
        self.state = 0


    def reward(self, state_p):
        return 1 if state_p == self.n_states-1 else 0
        

    def reset(self):
        self.state = 0
        return self.state


    def step(self, a):
        probs = np.zeros(self.n_states)
        for s_p in range(self.n_states):
            probs[s_p] = self.dynamics(s_p, a, self.state)

        self.state = np.random.choice(self.n_states, p=probs)
        return self.state


    def dynamics(self, s_p, a, s):
        x_a, y_a = self.actions[a]
        x, y = s%self.grid_size, s//self.grid_size
        x_p, y_p = s_p%self.grid_size, s_p//self.grid_size

        if not (0 <= x_p < self.grid_size and 0 <= y_p < self.grid_size):
            return 0.0
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return 0.0

        if abs(x-x_p) + abs(y-y_p) > 1:
            return 0.0

        if (x+x_a, y+y_a) == (x_p, y_p):
            return 1 - self.wind + self.wind/self.n_actions

        if (x, y) == (x_p, y_p):
            if not (0 <= x+x_a < self.grid_size and 0 <= y+y_a < self.grid_size):
                if (x,y) in [(0,0), (0,self.grid_size-1), (self.grid_size-1,0), (self.grid_size-1,self.grid_size-1)]:
                    return 1 - self.wind + 2*self.wind/self.n_actions
                else:
                    return 1 - self.wind + self.wind/self.n_actions
            else:
                if (x,y) in [(0,0), (0,self.grid_size-1), (self.grid_size-1,0), (self.grid_size-1,self.grid_size-1)]:
                    return 2 * self.wind/self.n_actions
                elif x == 0 or x == self.grid_size-1 or y == 0 or y == self.grid_size-1:
                    return self.wind/self.n_actions
                else:
                    return 0.0
       
        return self.wind/self.n_actions

    
    def test(self):
        for s in range(self.n_states):
            print("/// State: ", s)
            for a in range(self.n_actions):
                print("/// Action: ", self.names[a])
                probs = np.zeros((self.grid_size*self.grid_size))
                for s_p in range(self.n_states):
                    probs[s_p] = self.dynamics(s_p, a, s)
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