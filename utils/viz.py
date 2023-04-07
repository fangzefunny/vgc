import numpy as np 
import seaborn as sns 
import matplotlib.colors
import matplotlib as mpl 

class viz:
    '''Define the default visualize configure
    '''
    # platte one
    dBlue   = np.array([ 56,  56, 107]) / 255
    Blue    = np.array([ 46, 107, 149]) / 255
    lBlue   = np.array([241, 247, 248]) / 255
    lBlue2  = np.array([166, 201, 222]) / 255
    Green   = np.array([  8, 154, 133]) / 255
    lGreen  = np.array([242, 251, 238]) / 255
    dRed    = np.array([108,  14,  17]) / 255
    Red     = np.array([199, 111, 132]) / 255
    lRed    = np.array([253, 237, 237]) / 255
    lRed2   = np.array([254, 177, 175]) / 255
    dYellow = np.array([129, 119,  14]) / 255
    Yellow  = np.array([220, 175, 106]) / 255
    lYellow2= np.array([166, 201, 222]) / 255
    lYellow = np.array([252, 246, 238]) / 255
    Purple  = np.array([108,  92, 231]) / 255
    ocGreen = np.array([ 90, 196, 164]) / 255
    Palette = [Blue, Yellow, Red, ocGreen, Purple]

    # platte two
    dGreen  = np.array([ 15,  93,  81]) / 255
    llBlue  = np.array([118, 193, 202]) / 255
    Ebony   = np.array([ 86,  98,  70]) / 255
    deBlue  = np.array([ 66,  96, 118]) / 255
    fsGreen = np.array([ 79, 157, 105]) / 255
    Ercu    = np.array([190, 176, 137]) / 255
    ubSilk  = np.array([232, 204, 191]) / 255
    ppPant  = np.array([233, 214, 236]) / 255
    Palette2 = [dGreen, fsGreen, Ercu, ubSilk, ppPant]

    # platte three 
    dPurple = np.array([142,  65,  98]) / 255
    aPink   = np.array([237, 162, 192]) / 255
    oGrey   = np.array([176, 166, 183]) / 255
    black   = np.array([  0,   0,   0]) / 255
    Palette3 = [oGrey, aPink, dPurple]

    # Red gradient
    b1      = np.array([ 43, 126, 164]) / 255
    r1      = np.array([249, 199,  79]) / 255
    r2      = np.array([228, 149,  92]) / 255
    r3      = np.array([206,  98, 105]) / 255
    Palette4 = [b1, r1, r3] 

    Greens  = [np.array([  8, 154, 133]) / 255, 
               np.array([118, 193, 202]) / 255] 
    dpi     = 200
    sfz, mfz, lfz = 11, 13, 16
    lw, mz  = 2.5, 6.5
    figz    = 4

    BluePalette = [dBlue, Blue, lBlue]
    RedPalette  = [dRed, Red, lRed]
    YellowPalette = [dYellow, Yellow, lYellow]

    BluesMap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizBlues',   [lBlue, Blue])
    RedsMap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizReds',    [lRed, dRed])
    YellowsMap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizYellows', [lYellow, Yellow])
    GreensMap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizGreens',  [lGreen, Green])
    PurplesMap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizPurples', [np.clip(Purple*1.8, 0, 1), Purple])

    # for info plot 
    BluesMap2 = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizBlues',   [lBlue2, Blue])
    RedsMap2 = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizReds',    [lRed2, dRed])
    YellowsMap2 = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizYellows', [lYellow2, Yellow])
    
    # discrete 
    listmap = matplotlib.colors.ListedColormap([[1]*3, Blue, [0]*3])

    @staticmethod
    def get_style(): 
        # Larger scale for plots in notebooks
        sns.set_context('talk')
        sns.set_style("ticks", {'axes.grid': False})
        # mpl.rcParams['axes.spines.right']  = False
        # mpl.rcParams['axes.spines.top']    = False
