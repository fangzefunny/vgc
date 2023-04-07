import time 
import numpy as np 
from pynput import keyboard
from IPython.display import clear_output

import matplotlib.pyplot as plt 
import seaborn as sns 

from .viz import viz

emptygrid = [
        "..........G",
        "..1........",
        "..1........",
        "..1..#.....",
        ".....#.....",
        "...#####...",
        ".....#.....",
        ".....#.....",
        "...........",
        "...........",
        "S.........."
    ]

full_grid = [
    "G..........",
    ".......33..",
    "....4443...",
    "....4#.3.22",
    ".....#...2.",
    "555#####.2.",
    "5...0#.....",
    "....0#...66",
    "...00....6.",
    "1........6.",
    "111.......S"
]

full_grid2 = [
    "333.......S",
    "3..........",
    ".......111.",
    "4....#.1...",
    "4....#.....",
    "44.#####0..",
    ".....#6.000",
    ".....#6....",
    ".....66....",
    ".5......2..",
    "555..G222.."
]

key_map = {
    'w': 0,
    's': 1,
    'a': 2,
    'd': 3,
}

class girdmaze:

    def __init__(self, **kargs):

        # default dict
        self.config = {
            'layout': full_grid2,
            'construal': '0123456',
        }
        for k in kargs.keys(): self.config[k] = kargs[k]

        # init 
        self.direct = [
            np.array([-1, 0]), # up
            np.array([ 1, 0]), # down
            np.array([ 0,-1]), # left
            np.array([ 0, 1]), # right
        ]
        self.layout = self.config['layout']
        self.get_occupancy(self.config['construal'])
        self.nS = self.occupancy.reshape([-1]).shape[0]
        self.nA = 4
        self.state_space = np.arange(self.nS).astype(int)
        self.act_space   = np.arange(self.nA).astype(int)
        self.goal_state = self.cell2state(self.goal)
        self.state = self.cell2state(self.curr_cell)

    def cell2state(self, cell):
        return cell[0]*self.occupancy.shape[1] + cell[1]
    
    def state2cell(self, state):
        n = self.occupancy.shape[1]
        return np.array([state//n, state%n])
    
    def get_occupancy(self, config):
        map_dict = {
            '#': 1,
            '.': 0,
            'S': 0, 
            'G': 0,
        }
        for obst in range(10):
            map_dict[str(obst)] = 0 
        for obst in list(config):
            map_dict[obst] = .5
        self.occupancy = np.array([list(map(lambda x: map_dict[x], row)) 
                                   for row in self.layout])
        # get the goal loc
        self.goal = np.hstack(np.where(np.array([list(row) 
                        for row in self.layout])=='G'))
        self.curr_cell = np.hstack(np.where(np.array([list(row) 
                        for row in self.layout])=='S'))
        
    def get_p_trans(self):
        p_trans = {}
        for s in self.state_space:
            for a in self.act_space:
                p_trans[(s, a)] = self.phi(s, a)
        return p_trans
    
    def get_rew_fn(self):
        return np.array([(s==self.goal_state)-1 
                for s in self.state_space])
        
    def phi(self, state, act, eps=1e-5):
        if state == self.goal_state:
            return {state: 1}
        else:
            curr_cell = self.state2cell(state)
            if (self.occupancy[tuple(curr_cell)]==1):
                return {}
            else:
                next_cell = np.clip(curr_cell+self.direct[act], 0, 10)

                # φMove: pressing an arrow key causes the circle to move 
                # in that direction with probability 1 − ε and stay in place with probability ε, ε = 10−5), 
                
                # φWalls: the effect of being blocked by the centre, 
                # plus-shaped (+) walls (that is, the wall causes the circle to not move 
                # when the arrow key is pressed)

                # φObstaclei,i=1,..,.N: effects of being blocked by each of the 
                # N obstacles,.

                # if not blocked
                blocked = (self.occupancy[tuple(next_cell)]!=0)
                if blocked:
                    # stay 
                    return {state: 1}
                else:
                    # get next state 
                    next_state = self.cell2state(next_cell)
                    return {state: eps, next_state: 1-eps}
                

    # ------------ run the environment ----------- #

    def reset(self):
        self.done = False
        self.act = None

        return self.state, self.done 

    def render(self, ax):
        '''Visualize the figure
        '''
        occupancy = np.array(self.occupancy)
        sns.heatmap(occupancy, cmap=viz.listmap, ax=ax, 
                    lw=.5, linecolor=[.9]*3, cbar=False)
        ax.axhline(y=0, color='k',lw=5)
        ax.axhline(y=occupancy.shape[0], color='k',lw=5)
        ax.axvline(x=0, color='k',lw=5)
        ax.axvline(x=occupancy.shape[1], color='k',lw=5)
        ax.text(self.goal[1]+.15, self.goal[0]+.75, 'G', color=viz.Red,
                    fontweight='bold', fontsize=10)
        ax.text(self.curr_cell[1]+.25, self.curr_cell[0]+.75, 'O', color=viz.Red,
                    fontweight='bold', fontsize=10)
        ax.set_axis_off()
        ax.set_box_aspect(1)

    def check_available(self, cell):
        ava_cells = []
        for a in self.action_space:
            next_cell = tuple(cell + self.direct[a])
            if self.occupancy[next_cell] == 0:
                ava_cells.append(next_cell)
        return ava_cells
    
    def step(self, act):
        next_cell = np.clip(self.curr_cell+self.direct[act], 0, 10)
        if self.occupancy[tuple(next_cell)]==0:
            self.curr_cell = next_cell
        # check if the goal is reached
        self.state = self.cell2state(self.curr_cell)
        self.done  = self.state == self.goal_state
        rew = 0 if self.done else -1 
        self.act = None 
        return self.state, rew, self.done
    
    def on_press(self, key):
        try:
            self.act = key_map[key.char]
        except:
            self.act = None

    def on_release(self, key):
        if key == keyboard.Key.esc:
            # Stop listener
            return False

    def wait_until(self):
        while not self.act in [0, 1, 2, 3]:
            time.sleep(.1) 
        return self.act 
        
        
        
