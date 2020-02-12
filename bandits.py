import numpy as np
import matplotlib.pyplot as plt
from random import choice

# classes for bandits
# X_order_bandit is generated with either P or P_intervals. If P is set, then this bandit will 
# always have fixed probability P of correspondent's order switch 
# If P is not fixed, then probability is drawn from P_intervals each time bandit reset is called.
# During meta learning, reset is called each time episode finishe


# reset() -- resets bandit, draws new P from P_interval is P is not fixed
# update() -- updates timesteo and reward's position
# act() -- receives action, calls update, returns reward and information about trial 

def draw_from_intervals(intervals):
    chosen_interval = intervals[np.random.randint(len(intervals))]
    res = np.random.uniform(*chosen_interval)
    return res
def evaluate_Ps(history):
    history = np.array(history)    
    p_switch =  np.mean(history[1:] != history[:-1])
    return p_switch
def evaluate_all_probs(history):
    ''' Prints empirical probabilities'''
    history = np.array(history)
    
    p_one = np.mean(history)
    p_zero = 1 - p_one
    
    p_switch =  np.mean(history[1:] != history[:-1])
    p_stay = 1 - p_switch
    
    after_stay = np.array([history[i] != history[i - 1] for i in range(2, len(history)) if history[i - 1] == history[i - 2]])
    after_switch = np.array([history[i] != history[i - 1] for i in range(2, len(history)) if history[i - 1] != history[i - 2]])
    print('P(0) = %.2f, P(1) = %.2f' % (p_zero, p_one))
    print()
    print('P(switch) = %.2f, P(stay) = %.2f' % (p_switch, p_stay))
    print()
    print('P(switch|switch) = %.2f, P(stay|switch) = %.2f' %(np.mean(after_switch), 1 - np.mean(after_switch)))
    print('P(switch|stay) = %.2f, P(stay|stay) = %.2f' %(np.mean(after_stay), 1 - np.mean(after_stay)))
class zero_order_bandit():
    def __init__(self, P_intervals = [(0, .1), (0.2, 0.3), (0.7, 0.8), (0.9, 1.)], P = None):
        self.P_intervals = P_intervals
        self.actions = np.arange(0, 2, dtype='int')
        self.fix_P = False
        self.name = '0th_order'
        self.reset(P)
        
    def reset(self, P = None):
        self.prev_reward_pos = np.random.choice(self.actions,)
        self.prev_prev_reward_pos = np.random.choice(self.actions,)
        if P is not None:
            self.P = P
            self.fix_P = True
        else:
            if not self.fix_P:
                self.P = draw_from_intervals(self.P_intervals)
            
        self.best_arm = 0 if self.P >= 0.5 else 1
        #print('Best arm:', self.best_arm, 'P:', self.P)
        new_reward_pos = np.random.choice(self.actions, p = [self.P, 1- self.P])
        self.reward = [0, 0]
        self.reward[new_reward_pos] = 1
        self.reward_pos = new_reward_pos
        self.timestep = 0
        self.maxtimestep = 80
        
    def update(self):
        self.prev_prev_reward_pos = self.prev_reward_pos
        self.prev_reward_pos = self.reward_pos
        self.reward_pos = np.random.choice(self.actions, p = [self.P, 1- self.P])
        
        self.reward = [0, 0]
        self.reward[self.reward_pos] = 1
        self.timestep += 1
        
    def act(self, action):
        correct = 1 if action == self.best_arm else 0
        reward = self.reward[action]
        self.update()
        if self.timestep >= self.maxtimestep:
            d = True
        else:
            d = False
        return reward, d, self.timestep, correct
        
            
class first_order_bandit():
    def __init__(self, P_intervals = [(0, .1), (0.2, 0.3), (0.7, 0.8), (0.9, 1.)], P = None):
        self.P_intervals = P_intervals
        self.actions = np.arange(0, 2, dtype='int')
        self.fix_P = False
        self.name = '1st_order'
        self.reset(P)
        
    def reset(self, P = None):
        self.prev_reward_pos = np.random.choice(self.actions,)
        self.prev_prev_reward_pos = np.random.choice(self.actions,)
        if P is not None:
            self.P = P
            self.fix_P = True
        else:
            if not self.fix_P:
                self.P = draw_from_intervals(self.P_intervals)
        reward_pos = np.random.choice(self.actions,)
        self.reward = [0, 0]
        self.reward[reward_pos] = 1
        self.reward_pos = reward_pos
        self.timestep = 0
        self.maxtimestep = 80
        
    def update(self):
        self.prev_prev_reward_pos = self.prev_reward_pos
        self.prev_reward_pos = self.reward_pos
        if np.random.uniform() <= self.P:
            self.reward_pos =  1 - self.reward_pos
            self.reward = [0, 0]
            self.reward[self.reward_pos] = 1
        self.timestep += 1
        
    def act(self, action):
        if self.P >= .5:
            correct_action  = 1 - self.prev_reward_pos
        else:
            correct_action = self.prev_reward_pos
        correct = 1 if action == correct_action else 0
        reward = self.reward[action]
        self.update()
        if self.timestep >= self.maxtimestep:
            d = True
        else:
            d = False
        return reward, d, self.timestep, correct
 
class second_order_bandit():
    def __init__(self, P_intervals = [(0, .1), (0.2, 0.3), (0.7, 0.8), (0.9, 1.)], P = None):
        self.P_intervals = P_intervals
        self.actions = np.arange(0, 2, dtype='int')
        self.fix_P = False
        self.name = '2nd_order'
        self.reset(P)
        
    def reset(self, P = None):
        self.prev_reward_pos = np.random.choice(self.actions,)
        self.prev_prev_reward_pos = np.random.choice(self.actions,)
        if P is not None:
            self.P = P
            self.fix_P = True
        else:
            if not self.fix_P:
                self.P = draw_from_intervals(self.P_intervals)
        
        reward_pos = np.random.choice(self.actions,)
        self.reward = [0, 0]
        self.reward[reward_pos] = 1
        self.reward_pos = reward_pos
        self.timestep = 0
        self.maxtimestep = 80
        
    def update(self):
        if np.random.uniform() <= self.P:
            self.reward_pos, self.prev_reward_pos, self.prev_prev_reward_pos = 1 - self.prev_reward_pos, self.reward_pos, self.prev_reward_pos
        else:
            self.reward_pos, self.prev_reward_pos, self.prev_prev_reward_pos = self.prev_reward_pos, self.reward_pos, self.prev_reward_pos
            
        self.reward = [0, 0]
        self.reward[self.reward_pos] = 1
        self.timestep += 1
        
    def act(self, action):
        if self.P >= .5:
            correct_action  = 1 - self.prev_prev_reward_pos
        else:
            correct_action = self.prev_prev_reward_pos
        correct = 1 if action == correct_action else 0
        reward = self.reward[action]
        self.update()
        if self.timestep >= self.maxtimestep:
            d = True
        else:
            d = False
        return reward, d, self.timestep, correct

    
import matplotlib
def plot_history(hist, len_im = 20, len_box = 4, figsize = (12, 6)):
    im = []
    plt.figure(figsize = figsize)
    for i in range(len_im):
        val = hist[i]
        loc_im = 2 * val * np.ones((6, len_box))
        border = np.ones((6, 1))
        im.append(loc_im)
        im.append(border)
    im = np.concatenate(im, 1)
    ticks = list([0, 1, 2])
    ticks = [int(t) for t in ticks]
    cmap = plt.cm.hot
    norm = matplotlib.colors.BoundaryNorm(np.arange(min(ticks)- .5, max(ticks) + .6,1), cmap.N)
    plt.imshow(im, cmap = cmap)
    cbar = plt.colorbar(ticks = ticks)
    _ = cbar.ax.set_yticklabels(['State 0' , 'Boundary', 'State 1'])
    
class determenistic_bandit():
    def __init__(self, reward_seq,):
        self.reward_seq = reward_seq
        self.P = evaluate_Ps(reward_seq)
        self.reset()
        
    def reset(self):
        self.timestep = 0
        self.maxtimestep = len(self.reward_seq)
        
        self.reward_pos = self.reward_seq[self.timestep]
        self.reward = [0, 0]
        self.reward[self.reward_pos] = 1
        self.prev_reward_pos = self.reward_seq[self.timestep - 1]
        
    def update(self):
        self.timestep += 1
        self.prev_reward_pos = self.reward_pos
        if self.timestep >= self.maxtimestep:
            self.reward_pos = 0
        else:
            self.reward_pos =  self.reward_seq[self.timestep]
        self.reward = [0, 0]
        self.reward[self.reward_pos] = 1
        
    def act(self, action):
        if self.P >= .5:
            correct_action  = 1 - self.prev_reward_pos
        else:
            correct_action = self.prev_reward_pos
        correct = 1 if action == correct_action else 0
        reward = self.reward[action]
        self.update()
        if self.timestep >= self.maxtimestep:
            d = True
        else:
            d = False
        return reward, d, self.timestep, correct