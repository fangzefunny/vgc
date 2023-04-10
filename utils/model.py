import pickle
import time
import numpy as np 
from scipy.special import softmax

from functools import lru_cache
from tqdm import tqdm
from IPython.display import clear_output

import matplotlib.pyplot as plt
import seaborn as sns 

from .maze import girdmaze
from .viz import viz
from utils.auxiliary import *

# ------- Construal --------- #

class construal:

    def __init__(self, env_fn=girdmaze, config='', **kwargs):
        self.config = config
        self.env_fn = env_fn

        # load args
        for key, val in kwargs.items(): 
            setattr(self, key, val)

    def eval(self, gamma=.99):
       
        # solve the mdp using dynamic programming 
        env = self.env_fn(**{'config': self.config})
        _, policy_c = self.policy_iteration(env)
        cost = len(self.config)

        # evaluate the construal on the real environment
        # eq 2: U(π_c) = U(s_0) + \sum_a π(a|s0) \sum_s' p(s'|s,a)v_π(s') 
        real_env = self.env_fn(**{'config': '0123456'})
        value_r = self.policy_evaluation(real_env, policy_c, gamma)        
        s0, _ = real_env.reset()
        p_trans = real_env.get_p_trans()
        u_c = 0
        for a, pi in enumerate(policy_c[s0]):
            for s_next in p_trans[(s0, a)].keys():
                p_s_next = p_trans[(s0, a)][s_next]
                v_next   = value_r[s_next]
                u_c += gamma*pi*p_s_next*v_next
        vor = u_c - cost
        return {'v_c': value_r, 'c_c': cost, 'vor_c': vor, 'pi_c': policy_c}

    def policy_iteration(self, env, gamma=.99, tol=1e-6):
        '''Exactly solve the inner loop using
            dynamic programming method
        '''
        # initialize the π0
        policy = np.ones([env.nS, env.nA]) / env.nA 

        while True:
            # calculate the value of the policy
            value = self.policy_evaluation(env, policy, gamma, tol=tol)
            # improve the policy
            new_policy = self.policy_improvement(env, value, policy, gamma)
            # check convergence
            if (new_policy == policy).all():
                break
            # update and start the next iteration
            policy = new_policy.copy()

        return value, policy 
          
    def policy_evaluation(self, env, policy, gamma=.99, tol=1e-6):

        p_trans = env.get_p_trans()
        rew_fn  = env.get_rew_fn()
        value   = np.zeros([env.nS])
        
        while True:
            delta = 0 
            for s in env.state_space:
                v = 0 
                for a, pi in enumerate(policy[s]):
                    for s_next in p_trans[(s, a)].keys():
                        p_s_next = p_trans[(s, a)][s_next]
                        r        = rew_fn[s_next]
                        v += pi*p_s_next*(r + gamma*value[s_next])
                delta = np.max([delta, np.abs(value[s]-v)])
                value[s] = v
            if delta < tol:
                break 
        return value
    
    def policy_improvement(self, env, value, policy, gamma=.99):

        p_trans = env.get_p_trans()
        rew_fn  = env.get_rew_fn()
        
        for s in env.state_space:
            q = np.zeros([env.nA])
            for a in env.act_space:
                for s_next in p_trans[(s, a)].keys():
                    p_s_next = p_trans[(s, a)][s_next]
                    r        = rew_fn[s_next]
                    q[a] += p_s_next*(r + gamma*value[s_next])
            # argmax 
            a_max = np.argwhere(q==np.max(q)).flatten()
            policy[s] = np.sum([np.eye(env.nA)[i] for i in a_max], axis=0) / len(a_max)
        return policy 
    
    def render(self, env, pi, seed=1234, wait_time=.1, maxiter=50):
        rng = np.random.RandomState(seed)
        state, done = env.reset()
        occupancy = np.array(env.occupancy)
        traj = [env.curr_cell.copy()] 
        t = 0
        while True:
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            clear_output(True)
            act = rng.choice(env.nA, p=pi[state, :])
            state, _, done = env.step(act)
            sns.heatmap(occupancy, cmap=viz.mixMap, ax=ax, 
                        lw=.5, linecolor=[.9]*3, cbar=False)
            ax.axhline(y=0, color='k',lw=5)
            ax.axhline(y=occupancy.shape[0], color='k',lw=5)
            ax.axvline(x=0, color='k',lw=5)
            ax.axvline(x=occupancy.shape[1], color='k',lw=5)
            ax.text(env.goal[1]+.15, env.goal[0]+.75, 'G', color=viz.Red,
                        fontweight='bold', fontsize=10)
            ax.text(env.curr_cell[1]+.25, env.curr_cell[0]+.75, 'O', color=viz.Red,
                        fontweight='bold', fontsize=10)
            traj.append(env.curr_cell.copy())
            for t in range(len(traj)-1):
                ax.plot([traj[t][1]+.5, traj[t+1][1]+.5], 
                        [traj[t][0]+.5, traj[t+1][0]+.5], 
                        color=viz.Red)
            ax.set_axis_off()
            ax.set_box_aspect(1)
            time.sleep(wait_time)
            plt.show()
            t += 1
            if done or (t>maxiter): break

# --------- Baisc value guided construal --------- #

class baseVGC:

    def __init__(self, params, **kwargs):
        self.env_fn = girdmaze
        self.configs = get_all_configs()
        self.load_params(params)

        for key, val in kwargs.items():
            setattr(self, key, val)

    def load_params(self, params):
        self.alpha = params['alpha']
    
    def eval_consts(self):
        '''Prepare all possible construals
        '''

        all_c_info = {}
        for c_config in  tqdm(self.configs):
            c = construal(config=c_config)
            all_c_info[c_config] = c.eval()

        return all_c_info
    
    def load_consts(self, load_dir):
        with open(f'{load_dir}', 'rb')as handle:
            self.all_c_info = pickle.load(handle)
        self.policy_inner = {
            config: self.all_c_info['policy_c']
            for config in self.configs
        }

    def p_obst(self):
        '''Predict the memory of each obstacle
        '''
        # p_c = softmax()
        logit = np.array([self.all_c_info[c]['vor_c'] 
                for c in self.all_c_info.keys()])
        p_C = np.exp((logit - logit.max())/self.alpha)
        p_C = p_C / p_C.sum()
        
        # calculate the attention of
        # each obstacle
        N_c = np.zeros([7])
        for i, k in enumerate(self.all_c_info.keys()):
            for c in list(k):
                N_c[int(c)] += p_C[i] 

        return N_c / N_c.sum()
    

class dynamicVGC(baseVGC):
    
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.nS = self.env_fn.nS
        self.nC = len(self.configs)
        if not hasattr(self, 'seed'):
            self.seed = 1234
        self.rng = np.random.RandomState(self.seed)
        self.c = self.rng.choice(self.nC)

        self._init_value()

    @lru_cache()
    def get_trans_outer(self, s, const_next, p_trans):
        '''
        p(s'|s, c') = \sum_a π(a|s, c') p(s'|s, a)
        '''
        policy_c_next = self.policy_inner[const_next]
        p_trans_outer = {}
        for a, pi in enumerate(policy_c_next[s]):
            for s_next, p_s_next in p_trans[(s, a)].items():
                if s_next not in p_trans_outer.keys():
                    p_trans_outer[s_next] = 0
                p_trans_outer[s_next] += pi*p_s_next
        return p_trans_outer

    def get_value_outer(self, env, tol=1e-4):

        # get the real transition function
        p_trans = env.get_p_trans()
        rew_fn  = env.get_rew_fn()
        
        # initialize the V(s,c)
        value = np.zeros([self.nS, self.nC])

        # iterate to estimate the value 
        while True:
            
            new_value = np.zeros([self.nS, self.nC])
            for s in env.state_space:
                for c, const in enumerate(self.policy_inner.keys()):

                    q = np.zeros([self.nC])
                    for c_next, const_next in enumerate(self.policy_inner.keys()):
                        
                        # get outer-loop level transition
                        p_trans_outer = self.get_trans_outer(s, const_next, p_trans)
                        
                        # define the current value 
                        # q = \sum_s' p(s'|s,c')V(s',c') - C(c', c)
                        v_s_next = 0
                        for s_next, p_s_next in p_trans_outer.items():
                            v_s_next += p_s_next*value[s_next, c_next] 
                        cost_next = len(const_next) - len(const)
                        q[c_next] = v_s_next - cost_next

                    # v(s,c) = u(s) + \max_c'[q]
                    new_value[s, c] = rew_fn[s] + q.max()

            # check convergence
            if np.max(new_value - value) < tol: 
                break
            
            # update and start the next iteration
            value = new_value.copy()

        self.value = value

        return self.value
    
    def policy_outer(self, s, env):

        # get the transition function
        p_trans = env.get_p_trans()
        
        # get the current construal
        const = self.configs[self.c]
        
        q = np.zeros([self.nC])
        for c_next, const_next in enumerate(self.policy_inner.keys()):
            
            # get outer-loop level transition
            p_trans_outer = self.get_trans_outer(s, const_next, p_trans)
            
            # define the current value 
            # q = \sum_s' p(s'|s,c')V(s',c') - C(c', c)
            v_s_next = 0
            for s_next, p_s_next in p_trans_outer.items():
                v_s_next += p_s_next*self.value[s_next, c_next] 
            cost_next = len(const_next) - len(const)
            q[c_next] = v_s_next - cost_next

        return softmax(q/self.alpha)

        
    
