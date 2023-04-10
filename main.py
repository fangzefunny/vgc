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
from utils.model import *
from utils.viz import viz 
viz.get_style()

pth = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(f'{pth}/data'): os.mkdir(f'{pth}/data')


if __name__ == '__main__':

    model = baseVGC(params={'alpha': .1})

    # evaluate all construals
    if eval:
        all_c_info = model.eval_consts()
        with open(f'{pth}/data/all_c_info.pkl', 'wb')as handle:
            pickle.dump(all_c_info, handle)
        
    model.load_consts()