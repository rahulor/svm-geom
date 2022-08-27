import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 11,
})


def make_axis(ax, **params):
    d1, d2 = -1.0, 4.0
    Ticks = np.arange(d1,d2+0.1,1)
    ax.set_xticks(Ticks)
    ax.set_yticks(Ticks)
    ax.set_xlabel(r'$x$', size=18, labelpad=20)
    ax.set_ylabel(r'$y$', size=18,  rotation=0, labelpad=20)
    ax.axis('scaled')
    ax.set_xlim([d1,d2])
    ax.set_ylim([d1,d2])
    return ax

def dataset(filename):
    clist = plt.get_cmap('tab10')
    psize = 1.
    
    df = pd.read_csv(filename)
    fig, axis = plt.subplots(figsize=(5,3.5))
    ax = make_axis(axis)
    i = 0
    for name, grp in df.groupby('label'):
        ax.scatter(grp['x'], grp['y'], s=psize, color=clist(i), label=name)
        i+=1
    #plt.savefig('fig_data.jpg',bbox_inches='tight', dpi=200)
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1.03), title='label')
    plt.show()

def dataset3d(filename):
    clist = plt.get_cmap('tab10')
    psize = 1.
    
    df = pd.read_csv(filename)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    i = 0
    for name, grp in df.groupby('label'):
        ax.scatter(grp['x'], grp['y'], grp['z'], marker='o', s=psize, color=clist(i), label=name)
        i+=1
    ax.set_xlabel(r'$x$', size=18, labelpad=20)
    ax.set_ylabel(r'$y$', size=18,  rotation=0, labelpad=20)
    ax.set_zlabel(r'$z$', size=18,  rotation=0, labelpad=20)
    # ax.axis('scaled')

    plt.show()

if __name__ == "__main__":
    filenames = ["sparse.csv", "circles.csv", "lines.csv", "lines_horz.csv", "circles_nonsep.csv"]
    for filename in filenames:
        dataset(filename)
    
    dataset3d("clouds.csv")
