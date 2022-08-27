import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gsvm.hyperplane import HyperPlane
from gsvm.classifier import GeometricClassifier
hp = HyperPlane()

clist = plt.get_cmap('tab10')
fig_ext = '.png'
d1, d2 = -1.0, 4.0
def make_axis(ax, **params):
    d1, d2 = -1.0, 4.0
    Ticks = np.arange(d1,d2+0.1,1)
    ax.set_xticks(Ticks)
    ax.set_yticks(Ticks)
    ax.set_xlabel(r'$x$', size=18, labelpad=10)
    ax.set_ylabel(r'$y$', size=18,  rotation=0, labelpad=10)
    ax.axis('scaled')
    ax.set_xlim([d1,d2])
    ax.set_ylim([d1,d2])
    return ax
    
def make_meshgrid(h=0.01):
    x_min, x_max = d1, d2
    y_min, y_max = d1, d2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def draw_margin(df, SV, w0, b0, file):
    psize = 0.5
    fig, axis = plt.subplots(figsize=(3,3))
    ax = make_axis(axis)
    i = 0
    for name, grp in df.groupby('label'):
        ax.scatter(grp['x'], grp['y'], s=psize, color=clist(i), label=name)
        i+=1
    if w0 is not None: 
        xx, yy = make_meshgrid()
        Z = hp.predict(w0, b0, np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        z = hp.decision_function(w0, b0, np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=plt.cm.binary, alpha=0.2)
        ax.contour(xx, yy, z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    ax.scatter(SV[:, 0], SV[:, 1], s=100, linewidth=0.5, facecolors='none', edgecolors='black')
    figpath = 'doc/fig/' + file + '_margin' + fig_ext
    plt.savefig(figpath,bbox_inches='tight', dpi=200)
    plt.show()

def dataset(df, file):
    clist = plt.get_cmap('tab10')
    psize = 1.0
    
    fig, axis = plt.subplots(figsize=(5,3.5))
    ax = make_axis(axis)
    i = 0
    for name, grp in df.groupby('label'):
        ax.scatter(grp['x'], grp['y'], s=psize, color=clist(i), label=name)
        i+=1
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1.03), title='label')
    figpath = 'doc/fig/' + file + fig_ext
    plt.savefig(figpath,bbox_inches='tight', dpi=200)
    plt.show()

def dataset3d(df):
    clist = plt.get_cmap('tab10')
    psize = 1.
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
    figpath = 'doc/fig/cloud' + fig_ext
    plt.savefig(figpath,bbox_inches='tight', dpi=200)
    plt.show()
def three_D_data():
    df = pd.read_csv("data/clouds.csv")
    dataset3d(df)
    X = df[['x', 'y', 'z']].values
    label = df['label'].values
    clf = GeometricClassifier()
    clf.fit(X, label)
    print('-'*60)
    print('clouds')
    print(clf.message)
    if clf.success:
        print(f'width = {clf.width_}')
        print(f'support_vectors\n{clf.support_vectors_}')
    return
def two_D_data():
    # 2D data sample
    filenames = ["sparse", "circles", "lines", "lines_horz", "circles_nonsep"]
    for file in filenames:
        df = pd.read_csv("data/"+file+".csv")
        dataset(df, file)
        X = df[['x', 'y']].values
        label = df['label'].values    
        clf = GeometricClassifier()
        clf.fit(X, label)
        print('-'*60)
        print(file)
        print(clf.message)
        if clf.success:
            print(f'width = {clf.width_}')
            print(f'support_vectors\n{clf.support_vectors_}')
            draw_margin(df, clf.support_vectors_, clf.coef_, clf.intercept_, file)
    return
if __name__ == "__main__":
    two_D_data()
    three_D_data()