import os 
import time 
import itertools
import pickle
import numpy as np 

import matplotlib.pyplot as plt 
import seaborn as sns 

from tqdm import tqdm

from utils.auxiliary import *
from utils.maze import girdmaze
from utils.model import construal
from utils.viz import viz 
viz.get_style()

pth = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(f'{pth}/data'): os.mkdir(f'{pth}/data')

def eval_all_construals():
    all_configs = get_all_construals()
    all_c_info = {}
    for c_config in  tqdm(all_configs):
        c = construal(config=c_config)
        all_c_info[c_config] = c.eval()

    with open(f'{pth}/data/all_c_info.pkl', 'wb')as handle:
        pickle.dump(all_c_info, handle)


if __name__ == '__main__':

    eval_all_construals()