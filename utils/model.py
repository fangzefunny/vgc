import time
import numpy as np 
from IPython.display import clear_output

import matplotlib.pyplot as plt
import seaborn as sns 

from .maze import girdmaze
from .viz import viz


# ------- Construal --------- #

class construal:

    def __init__(self, env_fn=girdmaze, config=''):
        self.config = config
        self.env_fn = env_fn

    def eval(self, gamma=.99):
       
        # solve the mdp using dynamic programming 
        env = self.env_fn(**{'construal': self.config})
        _, policy_c = self.policy_iteration(env)
        cost = len(self.config)

        # evaluate the construal on the real environment
        # eq 2: U(π_c) = U(s_0) + \sum_a π(a|s0) \sum_s' p(s'|s,a)v_π(s') 
        real_env = self.env_fn(**{'construal': '0123456'})
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
            sns.heatmap(occupancy, cmap=viz.listmap, ax=ax, 
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