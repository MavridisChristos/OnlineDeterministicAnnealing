#!/usr/bin/env python
"""
Multi-Resolution (2) Binary Classification with underline Gaussian Distributions
Christos Mavridis & John Baras,
Electrical and Computer Engineering Dept.,
University of Maryland
"""

#%% Import Modules

import pickle
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

# plt.ioff() # turn off interactive mode: only show figures with plt.show()
plt.close('all')

#%% Distortion Functions

@njit(cache=True,nogil=True)
def _winner(y, datum, phi='phi_KL'):
    dists = np.zeros(len(y))
    for i in range(len(y)):
        dists[i]=_BregmanD(datum,y[i],phi)
    j = np.argmin(dists)
    return j, dists[j]

@njit(cache=True,nogil=True)
def _dot(x, y):
    # s = 0.0
    # for i in range(len(x)):
    #     s += x[i]*y[i]
    s = x*y
    return s

@njit(cache=True,nogil=True)
def _BregmanD(x, y, phi='phi_KL'):
    if phi == 'phi_Eucl':
        d = _dot(x-y,x-y)
    # elif phi == 'phi_KL':
    #     pzero = 1e-9
    #     logx = np.zeros_like(x)
    #     logy = np.zeros_like(y)
    #     sxy=0
    #     for i in range(len(x)):
    #         if x[i]<pzero:
    #             x[i]=pzero
    #         if y[i]<pzero:
    #             y[i]=pzero    
    #         logx[i] = np.log(x[i])
    #         logy[i] = np.log(y[i])
    #         sxy += x[i]-y[i]
    #     d = _dot(x,logx-logy) - sxy
    return d


def score(y, ylabels, data, labels, classification, Bregman_phi):
    d = 0.0
    for i in range(len(data)):
        if classification:
            d += datum_classification_error(y, ylabels, data[i], labels[i], Bregman_phi)
        else:
            d += datum_dissimilarity(y, ylabels, data[i], labels[i], Bregman_phi)
    return d/len(data) if classification else d      

def datum_dissimilarity(y, ylabels, datum, label, Bregman_phi):
 
    j,d = _winner(np.array(y), datum, Bregman_phi)
    
    return d   

def datum_classification_error(y, ylabels, datum, label, Bregman_phi):
    
    j,_ = _winner(np.array(y), datum, Bregman_phi)
    
    decision_label = ylabels[j]
    d = 0 if decision_label == label else 1
    return d      
        

#%%

# Number of samples per Gaussian
ns = 500

centers = [2,3,5,6,9]
sigmas = [0.15, 0.15, 0.15, 0.15, 0.15]
# sigmas = [0.35, 0.25, 0.15, 0.35, 0.25]
# sigmas = [0.20, 0.15, 0.10, 0.20, 0.15]

# Contour alpha
aa = 0.05
# samples alpha
aaa = 0.4

np.random.seed(0)
# np.random.seed(13)

#%% Sample Data Set

data0 = []
labels0 = []
data1 = []
labels1 = []
data = []
labels = []

# class o
cx = centers[0]
sx = sigmas[0]
for i in range(ns):
    data0.append( np.random.normal(cx,sx) )
    labels0.append(0)
    data.append( np.random.normal(cx,sx) )
    labels.append(0)
                
# class x
cx = centers[1]
sx = sigmas[1]
for i in range(ns):
    data1.append( np.random.normal(cx,sx) )
    labels1.append(1)
    data.append( np.random.normal(cx,sx) )
    labels.append(1)    
    
# class o
cx = centers[2]
sx = sigmas[2]
for i in range(ns):
    data0.append( np.random.normal(cx,sx) )
    labels0.append(0)
    data.append( np.random.normal(cx,sx) )
    labels.append(0)
                
# class x
cx = centers[3]
sx = sigmas[3]
for i in range(ns):
    data1.append( np.random.normal(cx,sx) )
    labels1.append(1)
    data.append( np.random.normal(cx,sx) )
    labels.append(1)  
    
# class o
cx = centers[4]
sx = sigmas[4]
for i in range(ns):
    data0.append( np.random.normal(cx,sx) )
    labels0.append(0)
    data.append( np.random.normal(cx,sx) )
    labels.append(0)
               
    
#%% Map to [0,1] and shuffle

# map to [0,1]
def maptoone(data,labels):
    train_min = np.min(np.min(data,0)) #-0.1
    train_max = np.max(np.max(data,0)) #+0.1
    train_domain = train_max-train_min 
    # add small margins so that we avoid 0.0 values (for KL divergence)
    train_min = train_min - 0.05*train_domain
    train_max = train_max + 0.05*train_domain
    train_domain = train_max-train_min 
    # transform
    data = (data-train_min)/train_domain
    # centers = (centers-train_min)/train_domain
    # sigmas = sigmas/train_domain
    
    
    # shuffle data
    shi = np.arange(len(data))
    np.random.shuffle(shi)
    
    data = [data[i] for i in shi]
    labels = [labels[i] for i in shi]

    return data,labels

data, labels = maptoone(data,labels)
data0, labels0 = maptoone(data0,labels0)
data1, labels1 = maptoone(data1,labels1)


#%% Plot Histogram

fig_size=(10, 6)
font_size = 32
label_size = 38
legend_size = 14
line_width = 10
marker_size = 12
# fill_size=10
line_alpha = 0.1
# txt_size = 32
# txt_x = 1.0
# txt_y = 0.03
# font_weight = 'bold'
clrs = ['k','b','m','c','chocolate','g','y']
mrkrs = ['o','v','^','<','>','s','p']


fig,ax = plt.subplots(figsize=(10, 6),tight_layout = {'pad': 1})

plt.title('Probability Density in different scales')

# ax.set_ylim(vs-1,v0+1)
# ax.set_xlim(x0-1,1.3*xs)

# plt.xticks([x0,xs],[r'$s_i$',r'$s_{det}^i$'],fontsize = font_size)
# ax.set_xlabel('(m)', fontsize = font_size)
# plt.yticks([vs,v0],[r'$v_{det}^i$',r'$v_i$'],fontsize = font_size)
# ax.set_ylabel('(m/s)', fontsize = font_size)

binsi = [5,10,25,50,100,200]
for i in range(len(binsi)):

    hist = np.histogram(data,bins=binsi[i])
    plt.plot(np.linspace(0,1,len(hist[0])),[float(h)/max(hist[0]) for h in hist[0]],label=f'#bins:{binsi[i]}', color=clrs[i],linestyle='solid', 
              linewidth=line_width*(1-i*0.15), markersize=marker_size,alpha=line_alpha*(i+1))
    
# plt.grid(color='gray', linestyle='-', linewidth=1, alpha = 0.1)
plt.legend(prop={'size': legend_size})


#%% 1 codevector

# ys = [[yx] for yx in np.linspace(0,1,100)]
# ylabelss = [[0] for i in ys] 
# labels = [0 for i in data]
# ds = []

# for i in range(len(ys)):
#     y = ys[i]
#     ylabels = ylabelss[i]
#     ds.append( score(y=y, ylabels=ylabels, data=data, labels=labels, classification=False, Bregman_phi='phi_Eucl') )

# plt.plot(ds)
        
#%% 2 codevectors

ny = 50
yxs = np.linspace(0,1,ny)
yys = np.linspace(0,1,ny)
X,Y=np.meshgrid(yxs,yys)
ys = [[yx,yy] for yx in yxs for yy in yys]
ylabelss = [[0,0] for i in ys] 
labels = [0 for i in data]
ds = []

for i in range(len(ys)):
    y = ys[i]
    ylabels = ylabelss[i]
    ds.append( score(y=y, ylabels=ylabels, data=data, labels=labels, classification=False, Bregman_phi='phi_Eucl') )
    
Z = np.array(ds).reshape((ny,ny))

# pic = plt.imshow(yplane,cmap='Greys')
# plt.colorbar(pic)


# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


#%% Plot Distortion

fig_size=(10, 6)
font_size = 32
label_size = 38
legend_size = 24
line_width = 12
marker_size = 12
# fill_size=10
line_alpha = 0.1
# txt_size = 32
# txt_x = 1.0
# txt_y = 0.03
# font_weight = 'bold'
clrs = ['k','b','m','g','c','chocolate','y']
mrkrs = ['o','v','^','<','>','s','p']


fig,ax = plt.subplots(figsize=(10, 6),tight_layout = {'pad': 1})

# ax.set_ylim(vs-1,v0+1)
# ax.set_xlim(x0-1,1.3*xs)

# plt.xticks([x0,xs],[r'$s_i$',r'$s_{det}^i$'],fontsize = font_size)
# ax.set_xlabel('(m)', fontsize = font_size)
# plt.yticks([vs,v0],[r'$v_{det}^i$',r'$v_i$'],fontsize = font_size)
# ax.set_ylabel('(m/s)', fontsize = font_size)

binsi = [5,10,25,50,100,200]
for i in range(len(binsi)):

    hist = np.histogram(data,bins=binsi[i])
    plt.plot(np.linspace(0,1,len(hist[0])),hist[0],label=f'#bins:{i}', color=clrs[i], marker=mrkrs[i],linestyle='solid', 
              linewidth=line_width, markersize=marker_size,alpha=line_alpha)
    
# plt.grid(color='gray', linestyle='-', linewidth=1, alpha = 0.1)
plt.legend(prop={'size': legend_size})

#%% Algorithm Plot

# Create new Figure
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     # xlim=(b-3, b+3), ylim=(b-3, b+3))
                     xlim=(0, 1), ylim=(0, 1))
# ax.grid(True)
plt.xticks([0,1],'')
plt.yticks([0,1],'')

# data in 2D space
x_plot = [data[i] for i in range(len(data)) if labels[i] == 0]
y_plot = [data[i] for i in range(len(data)) if labels[i] == 1]

## plot data 
ax.plot([x_plot[i][0] for i in range(len(x_plot))],
        [x_plot[i][1] for i in range(len(x_plot))],'k.',alpha=aaa)
ax.plot([y_plot[i][0] for i in range(len(y_plot))],
        [y_plot[i][1] for i in range(len(y_plot))],'r.',alpha=aaa)

## Contours
delta = 0.005
# xm = np.arange(b-3.0, b+3.0, delta)
# ym = np.arange(b-3.0, b+3.0, delta)
xm = np.arange(0.0, 1.0, delta)
ym = np.arange(0.0, 1.0, delta)
Xm, Ym = np.meshgrid(xm, ym)

test = gauss_dens((0.5,0.5),0.05,Xm,Ym)
# class o
cx = np.array(centers[0])
sx = sigmas[0]
Zm =  gauss_dens(cx,sx,Xm,Ym)
ax.contour(Xm, Ym, Zm, alpha=aa, colors='k', levels=[1e-9, 1e-3, 1e-1, 1])

# class o
cx = np.array(centers[1])
sx = sigmas[1]
Zm =  gauss_dens(cx,sx,Xm,Ym)
ax.contour(Xm, Ym, Zm, alpha=aa, colors='k', levels=[1e-9, 1e-3, 1e-1, 1])

# class o
cx = np.array(centers[2])
sx = sigmas[2]
Zm =  gauss_dens(cx,sx,Xm,Ym)
ax.contour(Xm, Ym, Zm, alpha=aa, colors='k', levels=[1e-9, 1e-3, 1e-1, 1])
        
# class x
cx = np.array(centers[3])
sx = sigmas[3]
Zm =  gauss_dens(cx,sx,Xm,Ym)
ax.contour(Xm, Ym, Zm, alpha=aa, colors='r', levels=[1e-9, 1e-3, 1e-1, 1])

# class x
cx = np.array(centers[4])
sx = sigmas[4]
Zm =  gauss_dens(cx,sx,Xm,Ym)
ax.contour(Xm, Ym, Zm, alpha=aa, colors='r', levels=[1e-9, 1e-3, 1e-1, 1])
    

plt.show()
    
#%% Save Data

def proj(x): 
    u = np.array([1,1])
    u = u/np.linalg.norm(u)
    x = np.array(x)
    return np.dot(u,x)

# def proj_back(x):
#     u = np.array([1,1])
#     u = u/np.linalg.norm(u)
#     x = np.array(x)
#     return u*x

train_data = [[np.array([proj(td)]),td] for td in train_data]
test_data = [[np.array([proj(td)]),td] for td in test_data]
    

# For high Resolution only:
# train_data = [[np.array(td)] for td in train_data]
# test_data = [[np.array(td)] for td in test_data]

# Save results to file 
mydata = [train_data,train_labels,test_data,test_labels]
            
with open(save_file, mode='wb') as file:
    pickle.dump(mydata, file) 
   
    