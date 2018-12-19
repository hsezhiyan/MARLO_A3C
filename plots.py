#import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def create_reward_curve(list_reward, display=False):
    plt.figure(1)
    x_len = len(list_reward)
    t = np.arange(0, x_len)
    plt.plot(t, list_reward)
    if display == True:
        plt.show()
    plt.savefig('results/reward_curve.png', bbox_inches='tight')
