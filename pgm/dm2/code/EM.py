
# coding: utf-8

import numpy as np
from scipy.stats import multivariate_normal

run Kmeans.py

# Loading data
data = np.loadtxt('../classification_data_HWK2/EMGaussian.data')

# Setting number of clusters and initialization with Kmeans
K = 4
assignments, means = kmeans(data, K, 1)


LABEL_COLOR_MAP = {0 : 'r',1 : 'b', 2: 'g',3:'c',4:'m',5:'y',6:'k'}
colors = [LABEL_COLOR_MAP[k] for k in assignments]

# Plot Kmeans result
plt.figure(figsize=(10,10))
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.scatter(data[:,0],data[:,1],color = colors,marker='+')
plt.scatter(means[:,0],means[:,1], marker=">")

##

n = data.shape[0]
d = data.shape[1]

## For (a), proportional to identity cov matrix
#lam = np.random.rand(K)
#cov_mat = [lam[k]*np.eye(d) for k in range(0,K)]

## For (b)
cov_mat = [np.cov(data.T)]*K

q = np.ones((n,K))
pi = np.random.rand(K)
pi = pi*1/sum(pi)
previous_means = means+10   

## EM

# While no convergence
while np.linalg.norm(means-previous_means)>0.00001:
    
    # Expectation Step
    for i in range(0,n):
        S = sum([pi[u]*multivariate_normal.pdf(data[i], mean=means[u], cov=cov_mat[u]) for u in range(0,K)])
        for k in range(0,K):
            q[i,k] = pi[k]*multivariate_normal.pdf(data[i], mean=means[k], cov=cov_mat[k])/S

    # Maximization step
    SUM_Q = sum([q[i,j] for i in range(0,n) for j in range(0,K)])
    cov_mat = []
    
    for k in range(0,K):

        S = sum([q[i,k] for i in range(0,n)])
        
        # covmat update
        diff = [data[i] - means[k] for i in range(0,n)]
        [v.resize(v.shape[0],1) for v in diff]

        ## For (a)
        #lam[k] = sum([np.dot(v.T,v) for v in diff])*1./(K*S)
        #cov_mat = [lam[k]*np.eye(d) for k in range(0,K)]

        ## For (b)
        cov_mat.append(sum([np.dot(v,v.T) for v in diff])*1./(S))
        
        # Means update
        previous_means = means
        means[k] = sum([data[i]*q[i,k] for i in range(0,n)])/S
       
        # pi update
        pi[k] = S/SUM_Q
        pi = pi*1./sum(pi) # Renormalization for computation errors
        


# Computes the assingements to cluster
p = np.ones((n,K))
w = np.ones((n,K))
for i in range(0,n):
    p[i,:] = [multivariate_normal.pdf(data[i], mean=means[k], cov=cov_mat[k]) for k in range(0,K)]

for i in range(0,n):
    w[i,:] = [pi[k]*p[i,k]/(sum(pi*p[i,:])) for k in range(0,K)]

assig = [np.argmax(w[i,:]) for i in range(0,n)]


## Plotting utilities, the function were extracted from online code
from matplotlib.patches import Ellipse

def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ellip.set_alpha(0.3)
    ax.add_artist(ellip)
    return ellip



# Plotting data and ellipses
plt.figure(figsize=(10,10))

plt.xlim(-15, 15)
plt.ylim(-15, 15)

plt.gca().set_aspect('equal', adjustable='box')

a = 0.2
plot_cov_ellipse(cov_mat[0],means[0],a)
plot_cov_ellipse(cov_mat[1],means[1],a)
plot_cov_ellipse(cov_mat[2],means[2],a)
plot_cov_ellipse(cov_mat[3],means[3],a)

plt.scatter(data[:,0],data[:,1],color = colors,marker='+')
plt.scatter(means[:,0],means[:,1], marker=">",color = 'y')


plt.show()





