# to create sample data set
# easy to run and check

import numpy as np
import pandas as pd
def create_small_data(filename):
    xy = np.array([
    [0,1],
    [1,1],
    [2,0.5],
    [0,0],
    [1,0],
    [3,2],
    [2,2],
    [1,3],
    [2,3],
    [3,2.5]
    ])
    label = ['flower', 'flower', 'flower', 'flower', 'flower', 'fruit', 'fruit', 'fruit', 'fruit', 'fruit']
    df = pd.DataFrame(xy, columns=['x', 'y'])
    df['label'] = label
    #df['z'] = np.ones(len(label))
    df = df.sample(frac=1).reset_index(drop=True) # return all rows (in random order). Drop old indices
    df.to_csv(filename, index=False)

def create_two_circles(filename):
    n_points = 200
    radius = 1.0
    dim = 2 
    c1 = np.array([0.5]*dim)
    c2 = np.array([2.5]*dim)
    r = np.empty((n_points, dim))
    theta = np.random.uniform(0, 4*np.pi, size=int(n_points))
    r[:,0], r[:,1] = radius*np.cos(theta), radius*np.sin(theta)
    circle1 = c1 + r
    circle2 = c2 + r
    label = ['alpha']*n_points + ['beta']*n_points 
    circles = np.vstack((circle1, circle2))
    df = pd.DataFrame(circles, columns=['x', 'y'])
    df['label'] = label
    #df['z'] = np.ones(len(label))*5
    df = df.sample(frac=1).reset_index(drop=True) # return all rows (in random order). Drop old indices
    df.to_csv(filename, index=False)
def create_nonsep_circles(filename):
    n_points = 200
    radius = 1.5
    dim = 2 
    c1 = np.array([0.5]*dim)
    c2 = np.array([2.5]*dim)
    r = np.empty((n_points, dim))
    theta = np.random.uniform(0, 4*np.pi, size=int(n_points))
    r[:,0], r[:,1] = radius*np.cos(theta), radius*np.sin(theta)
    circle1 = c1 + r
    circle2 = c2 + r
    label = ['alpha']*n_points + ['beta']*n_points 
    circles = np.vstack((circle1, circle2))
    df = pd.DataFrame(circles, columns=['x', 'y'])
    df['label'] = label
    #df['z'] = np.ones(len(label))*5
    df = df.sample(frac=1).reset_index(drop=True) # return all rows (in random order). Drop old indices
    df.to_csv(filename, index=False)
def create_two_lines(filename):
    n_points = 10
    liney = np.linspace(0, 3, n_points)
    linea = np.vstack((np.ones(n_points)*0.0, liney)).T
    lineb = np.vstack((np.ones(n_points)*3.0, liney)).T
    label = ['alpha']*n_points + ['beta']*n_points 
    
    n_points = 100
    liney = np.linspace(0, 3, n_points)
    linec = np.vstack((np.ones(n_points)*-0.5, liney)).T
    lined = np.vstack((np.ones(n_points)*3.5, liney)).T
    label = label +  ['alpha']*n_points + ['beta']*n_points 

    lines = np.vstack((linea, lineb, linec, lined))
    df = pd.DataFrame(lines, columns=['x', 'y'])
    df['label'] = label
    df = df.sample(frac=1).reset_index(drop=True) # return all rows (in random order). Drop old indices
    df.to_csv(filename, index=False)
def create_two_lines_horz(filename):
    n_points = 10
    linex = np.linspace(-1, 1, n_points)
    line_mid = np.vstack((linex, np.ones(n_points)*1.0)).T
    n_points = 100
    linex = np.linspace(-1, 0.5, n_points)
    line_top = np.vstack((linex, np.ones(n_points)*2.0)).T
    line_bot = np.vstack((linex, np.ones(n_points)*0.0)).T
    line_alpha = np.vstack((line_mid, line_top, line_bot))
    label_alpha = ['alpha']*210
    
    n_points = 10
    linex = np.linspace(2.5, 4, n_points)
    line_mid = np.vstack((linex, np.ones(n_points)*1.0)).T
    n_points = 100
    linex = np.linspace(2.5, 4, n_points)
    line_top = np.vstack((linex, np.ones(n_points)*2.0)).T
    line_bot = np.vstack((linex, np.ones(n_points)*0.0)).T
    line_beta = np.vstack((line_mid, line_top, line_bot))
    label_beta = ['beta']*210
    
    lines = np.vstack((line_alpha, line_beta))
    labels = label_alpha + label_beta
    
    
    df = pd.DataFrame(lines, columns=['x', 'y'])
    df['label'] = labels
    df = df.sample(frac=1).reset_index(drop=True) # return all rows (in random order). Drop old indices
    df.to_csv(filename, index=False)
    
def multidim_cloud(filename):
    n_points = 1000
    dim = 3
    cloud_alpha = np.random.uniform(low=0.3, high=2, size=(n_points,dim))
    cloud_beta  = np.random.uniform(low=-0.3, high=-2, size=(n_points,dim))
    label_alpha = ['alpha']*n_points
    label_beta  = ['beta']*n_points
    clouds = np.vstack((cloud_alpha, cloud_beta))
    labels = label_alpha + label_beta
    df = pd.DataFrame(clouds, columns=['x', 'y', 'z'])
    df['label'] = labels
    df = df.sample(frac=1).reset_index(drop=True) # return all rows (in random order). Drop old indices
    df.to_csv(filename, index=False)
    
if __name__ == "__main__":
    create_small_data("sparse.csv")
    create_two_circles("circles.csv")
    create_two_lines("lines.csv")
    create_two_lines_horz("lines_horz.csv")
    create_nonsep_circles("circles_nonsep.csv")
    multidim_cloud("clouds.csv")
    