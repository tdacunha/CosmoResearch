#!/usr/bin/env python
# coding: utf-8

# # Parameter covariance (or lack thereof) of PCA and KL decompositions

# In[2]:

#
# # Show plots inline, and load main getdist plot module and samples class
# #Change
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '1')
# import libraries:
import sys, os
# here = './'
# temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
from getdist import plots, MCSamples
from getdist.gaussian_mixtures import GaussianND
import getdist
getdist.chains.print_load_details = False
import scipy
import matplotlib.pyplot as plt
import IPython
from IPython.display import Markdown
import numpy as np
import seaborn as sns
# import the tensiometer tools that we need:
# from tensiometer import utilities

here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)
from tensiometer import utilities
from tensiometer import gaussian_tension


# In[141]:


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],})


# In[17]:


# load the chains (remove no burn in since the example chains have already been cleaned):
chains_dir = here+'/tensiometer/test_chains/'
# the data chain:
settings = {'ignore_rows':0, 'smooth_scale_1D':0.3, 'smooth_scale_2D':0.3}
chain = getdist.mcsamples.loadMCSamples(file_root=chains_dir+'DES', no_cache=True, settings=settings)
#chain = getdist.mcsamples.loadMCSamples(file_root=chains_dir+'Planck18TTTEEE', no_cache=True, settings=settings)
# the prior chain:
prior_chain = getdist.mcsamples.loadMCSamples(file_root=chains_dir+'prior', no_cache=True, settings=settings)


# In[18]:


# add useful parameters:
for ch in [chain, prior_chain]:
    p = ch.getParams()
    ch.addDerived(p.omegach2/(p.H0/100.)**2, name='omegac', label='\\Omega_c')
    ch.addDerived(p.omegabh2/(p.H0/100.)**2, name='omegab', label='\\Omega_b')
    ch.addDerived(p.sigma8*np.sqrt(p.omegam/0.3), name='S8', label='S_8')
    ch.addDerived(p.sigma8*5, name='sigma8_5', label='\\sigma_8 5')
    ch.addDerived(p.H0/100, name='h', label='h')
    # update after adding all parameters:
    ch.updateBaseStatistics()


# In[19]:


# decide the parameters to use:
#param_names = ['omegam', 'omegab', 'sigma8', 'ns', 'H0']
param_names = ['omegam', 'sigma8']


# In[20]:


# add log of the chosen parameters:
for ch in [chain, prior_chain]:
    for name in param_names:
        ch.addDerived(np.log(ch.samples[:, ch.index[name]]), name='log_'+name, label='\\log'+ch.getParamNames().parWithName(name).label)
    # update after adding all parameters:
    ch.updateBaseStatistics()


# In[21]:


# now we generate the Gaussian approximation:
g_param_names = ['log_'+name for name in param_names]
n_samples = 100000
print(g_param_names)
# create the Gaussian:
gauss = gaussian_tension.gaussian_approximation(chain, param_names=g_param_names)
chain_gauss = gauss.MCSamples(size = n_samples)
prior_gauss = gaussian_tension.gaussian_approximation(prior_chain, param_names=g_param_names)
prior_chain_gauss = prior_gauss.MCSamples(size = n_samples)


# In[22]:


# look qualitatively at how the Gaussian approximation is doing:
g = plots.get_subplot_plotter()
g.triangle_plot([chain_gauss, chain], params=g_param_names, filled=True)


# In[23]:


# after generating the two perfectly Gaussian chains we can compute here the other parameters as derived:
for ch in [chain_gauss, prior_chain_gauss]:
    p = ch.getParams()
    # the original parameters:
    for ind, name in enumerate(g_param_names):
        ch.addDerived(np.exp(ch.samples[:, ch.index[name]]), name=str(name).replace('log_',''), label=str(ch.getParamNames().parWithName(name).label).replace('\\log',''))
    # label=ch.getParamNames().parWithName(name).label.replace('\\log ','')
    ch.updateBaseStatistics()


# In[24]:


print(chain_gauss.getParamNames().labels())


# In[25]:


# define the parameter transformation:
A = np.array([[ 1, -1, 0, 0, 0],
              [ 0, 1, 0, 0, 0],
              [ 0, 0, 1, 0, 0],
              [ 0, 0, 0, 1, 0],
              [ 0, 0, 0, 0, 1]])

A = np.array([[ 1, -1],
              [ 0, 1]])
# In[26]:


# verify that it is not orthogonal:
np.linalg.inv(A)


# In[27]:


theta = np.array([1,.5,0,0,0])
theta_prime = np.dot(A,theta)
print(theta_prime)


# In[28]:


covariance = chain.cov(pars=g_param_names)
# not using localization:
prior_covariance = prior_chain.cov(pars=g_param_names)
# using localization:
#prior_covariance = gaussian_tension.get_localized_covariance(prior_chain, chain, g_param_names)
tilde_covariance = np.dot(np.dot(A, covariance), A.T)
tilde_prior_covariance = np.dot(np.dot(A, prior_covariance), A.T)


# In[29]:


# do PCA of the two covariances:
PCA_eig, PCA_eigv = np.linalg.eigh(covariance)
tilde_PCA_eig, tilde_PCA_eigv = np.linalg.eigh(tilde_covariance)


# In[30]:


# do KL of the two covariances:
KL_eig, KL_eigv = utilities.KL_decomposition(prior_covariance, covariance)
tilde_KL_eig, tilde_KL_eigv = utilities.KL_decomposition(tilde_prior_covariance, tilde_covariance)

print(chain.cov(pars=param_names))
print(KL_eig, KL_eigv)
cov1 = chain.cov(pars=param_names)
prior_cov1 = prior_chain.cov(pars=param_names)
KL_eig1, KL_eigv1 = utilities.KL_decomposition(prior_cov1, cov1)
print(KL_eig1, KL_eigv1)

param_directions = np.linalg.inv(KL_eigv.T)
print(param_directions)
inds = (np.argsort(KL_eig)[::-1])
param_directions_best = ((param_directions.T[inds]).T)#[:,:2]
print(param_directions_best)
plt.figure()
m1,m2= chain_gauss.getMeans(pars=[chain_gauss.index[name]
               for name in [log_param_names_plot[0], log_param_names_plot[1]]])#np.log10(.25),np.log10(.9)
mode = 0
alpha = 100.*np.linspace(-1.,1.,10000) # 3
plt.plot(np.exp(m1+alpha*param_directions_best[0,mode]), np.exp(m2+alpha*param_directions_best[1,mode]), color='firebrick', lw=1.5, ls='-', marker = 'o',label='KL flow covariance')
#ax1.plot(maximum_posterior[0]+alpha*eigv[0,mode]/eig[mode], maximum_posterior[1]+alpha*eigv[1,mode]/eig[mode], lw=1.5, color='firebrick', ls='-', label='KL flow covariance')
#ax1.axline(maximum_posterior, maximum_posterior+eig[mode]*eigv[:, mode], lw=1., color=color_utilities.nice_colors(1), ls='-')
mode = 1

plt.plot(np.exp(m1+alpha*param_directions_best[0,mode]), np.exp(m2+alpha*param_directions_best[1,mode]), lw=1.5, color='cadetblue', ls='-', marker = 'o')
plt.xlim(0,.5)
plt.ylim(.4,1.4)
plt.show()



param_directions = np.linalg.inv(KL_eigv1.T)
print(param_directions)
inds = (np.argsort(KL_eig1)[::-1])
param_directions_best = ((param_directions.T[inds]).T)#[:,:2]
print(param_directions_best)


plt.figure()
m1,m2= (.25),(.9)
mode = 0
alpha = 100.*np.linspace(-1.,1.,10000) # 3
plt.plot((m1+alpha*param_directions_best[0,mode]), (m2+alpha*param_directions_best[1,mode]), color='firebrick', lw=1.5, ls='-', marker = 'o',label='KL flow covariance')
#ax1.plot(maximum_posterior[0]+alpha*eigv[0,mode]/eig[mode], maximum_posterior[1]+alpha*eigv[1,mode]/eig[mode], lw=1.5, color='firebrick', ls='-', label='KL flow covariance')
#ax1.axline(maximum_posterior, maximum_posterior+eig[mode]*eigv[:, mode], lw=1., color=color_utilities.nice_colors(1), ls='-')
mode = 1

plt.plot((m1+alpha*param_directions_best[0,mode]), (m2+alpha*param_directions_best[1,mode]), lw=1.5, color='cadetblue', ls='-', marker = 'o')
plt.xlim(0,.5)
plt.ylim(.4,1.4)
plt.show()



# In[31]:


# transform the PCA modes in the original basis:
PCA_eigv_trans =  np.dot(A.T, tilde_PCA_eigv)
PCA_eig_trans =  np.dot(A.T, tilde_PCA_eig)

print(PCA_eigv_trans)
print(PCA_eig_trans)


# In[32]:



print(PCA_eigv)
print(PCA_eig)


# In[33]:


# transform the KL modes in the original basis:
KL_eigv_trans = np.dot(A.T, tilde_KL_eigv)
KL_eig_trans =  np.dot(A.T, tilde_KL_eig)

print(KL_eigv_trans)
print(KL_eig_trans)


# In[34]:


KL_eigv


# In[35]:


print(chain_gauss.getParamNames().labels())


# In[142]:


KL_title_list = ['KL','KL transformed']
KL_eigv_list = [KL_eigv, KL_eigv_trans]
KL_eig_list = [KL_eig, KL_eig_trans]

PCA_title_list = ['PCA','PCA transformed']
PCA_eigv_list = [PCA_eigv, PCA_eigv_trans]
PCA_eig_list = [PCA_eig, PCA_eig_trans]

ls_list = ['-','--']

nmodes = 2

param_names_plot = ['omegam','sigma8']
log_param_names_plot = ['log_'+name for name in param_names_plot]

g = plots.get_subplot_plotter(subplot_size = 5)

g.plot_2d([chain_gauss], param1=param_names_plot[0], param2 = param_names_plot[1], filled=True, colors = ['darkcyan','darkcyan'])
ax1 = g.subplots[0,0]
alpha = 100*np.linspace(-1, 1, 1000)
i = 0
eig, eigv = KL_eig, KL_eigv
param_directions = np.linalg.inv(eigv.T)
inds = (np.argsort(eig)[::-1])
param_directions_best = ((param_directions.T[inds]).T)#[:,:2]
m1, m2 = np.log10(.025),np.log10(.9)#chain_gauss.getMeans(pars=[chain_gauss.index[name]
               #for name in [log_param_names_plot[i], log_param_names_plot[j]]])

print(np.shape(param_directions_best))
mode = 0
norm0 = np.linalg.norm(eigv[:,0])
ax1.plot(np.exp(m1+alpha*param_directions_best[0,mode]), np.exp(m2+alpha*param_directions_best[1,mode]), color='firebrick', lw=1.5, ls='-', marker = 'o',label='KL flow covariance')
#ax1.plot(maximum_posterior[0]+alpha*eigv[0,mode]/eig[mode], maximum_posterior[1]+alpha*eigv[1,mode]/eig[mode], lw=1.5, color='firebrick', ls='-', label='KL flow covariance')
#ax1.axline(maximum_posterior, maximum_posterior+eig[mode]*eigv[:, mode], lw=1., color=color_utilities.nice_colors(1), ls='-')
#print(np.log10(m1),np.log10(m2))
mode = 1
norm1 = np.linalg.norm(eigv[:,1])
ax1.plot(np.exp(m1+alpha*param_directions_best[0,mode]), np.exp(m2+alpha*param_directions_best[1,mode]), lw=1.5, color='cadetblue', ls='-', marker = 'o')
plt.show()


for title_list, eigv_list in zip([KL_title_list, PCA_title_list], [KL_eigv_list, PCA_eigv_list]):
    g = plots.get_subplot_plotter(subplot_size = 5)
    #g.triangle_plot([chain_gauss], params=param_names_plot, filled=True)
    g.plot_2d([chain_gauss], param1=param_names_plot[0], param2 = param_names_plot[1], filled=True, colors = ['darkcyan','darkcyan'])

    for n in range(0,2):
        title = title_list[n]
        eigv = eigv_list[n]

        ls = ls_list[n]

        param_directions = np.linalg.inv(eigv.T)

        # add the modes:
        for i in range(len(param_names_plot)-1):
            for j in range(i+1,len(param_names_plot)):
                ax = g.subplots[i,i]
                #ax = g.plots
                # get mean:
                m1, m2 = chain_gauss.getMeans(pars=[chain_gauss.index[name]
                               for name in [log_param_names_plot[i], log_param_names_plot[j]]])
                ax.scatter(np.exp(m1), np.exp(m2), color='k')


                list_ind = [chain_gauss.index[name]
                               for name in [log_param_names_plot[i], log_param_names_plot[j]]]
                i1 = list_ind[0]
                j1 = list_ind[1]
                if 'PCA' in title:
                    eig = PCA_eig_list[n]
                    inds = np.argsort(eig)
                    alpha = 10.*np.linspace(-1.,1.,1000) # 3
                else:
                    eig = KL_eig_list[n]
                    inds = (np.argsort(eig)[::-1])
                    alpha = 100.*np.linspace(-1.,1.,10000) # 3


                palette_l = [.5,.75]
                param_directions_best = ((param_directions.T[inds]).T)[:,:nmodes]
                for k in range(0,nmodes):

#                     if'PCA' in title:
#                         param_mag = (eig.T)
#                         factor = np.sqrt(param_mag[k])
#                         #alpha *= factor
#                         param_directions *= np.tile(param_mag.T,np.array([5,1]))
#                         print(np.shape(param_directions), np.shape(param_mag))
#                         #print(alpha)

                    ax.plot(np.exp(m1+alpha*(param_directions_best[:,k][i1])), np.exp(m2+alpha*(param_directions_best[:,k][j1])), color=sns.hls_palette(9,l = palette_l[n])[k], label=title+' mode '+str(k+1), ls=ls)

    #g.fig.legend(*ax.get_legend_handles_labels())
    g.fig.legend(loc = (.51,.77))




# # Running with Gaussian chains

# In[ ]:


param_names_gauss = ['omegam', 'omegab', 'sigma8', 'ns', 'H0']
print(param_names_gauss)
KL_param_names_gauss = ['log_'+name for name in param_names_gauss]
print(KL_param_names_gauss)
# compute the KL modes:
KL_eig, KL_eigv, KL_param_names_gauss = gaussian_tension.Q_UDM_KL_components(prior_chain_gauss, chain_gauss, param_names=KL_param_names_gauss)
# print:
with np.printoptions(precision=2, suppress=True):
    print('Improvement factor over the prior:', KL_eig)
    print('Improvement in error units:', np.sqrt(KL_eig-1))

# do the PCA on the log parameters:
#print(chain)

PCA_param_names = ['log_'+name for name in param_names_gauss]
# compute the PCA modes:
#PCA_eig, PCA_eigv, PCA_param_names =
#PCA_out = chain_gauss.PCA(KL_param_names)
#print(PCA_out)

# print:
#with np.printoptions(precision=2, suppress=True):
#    print('Improvement factor over the prior:', PCA_eig)
#    print('Improvement in error units:', np.sqrt(PCA_eig-1))


# In[ ]:


PCA_eig, PCA_eigv = np.linalg.eigh(chain_gauss.cov(KL_param_names_gauss))
print(PCA_eig)


# In[ ]:


# First we compute the fractional Fisher matrix:
KL_param_names, KL_eig, fractional_fisher, _ = gaussian_tension.Q_UDM_fisher_components(prior_chain, chain, KL_param_names, which='1')
# plot (showing values and names):
im1 = plt.imshow( fractional_fisher, cmap='viridis')
num_params = len(fractional_fisher)
for i in range(num_params):
    for j in range(num_params):
        if fractional_fisher[j,i]>0.5:
            col = 'k'
        else:
            col = 'w'
        plt.text(i, j, np.round(fractional_fisher[j,i],2), va='center', ha='center', color=col)
plt.xlabel('KL mode (error improvement)');
plt.ylabel('Parameters');
ticks  = np.arange(num_params)
labels = [ str(t+1)+'\n ('+str(l)+')' for t,l in zip(ticks,np.round(np.sqrt(KL_eig-1.),2))]
plt.xticks(ticks, labels, horizontalalignment='center');
labels = [ '$'+chain.getParamNames().parWithName(name).label+'$' for name in KL_param_names ]
plt.yticks(ticks, labels, horizontalalignment='right');


# In[ ]:


for eigv,title in zip([KL_eigv, PCA_eigv], ['KL','PCA']):
    param_names_plot = ['omegam','sigma8']
    KL_param_names_plot = ['log_'+name for name in param_names_plot]

    param_directions = np.linalg.inv(eigv.T)
    g = plots.get_subplot_plotter()
    g.triangle_plot([chain_gauss], params=param_names_plot, filled=True)
    # add the modes:
    for i in range(len(param_names_plot)-1):
        for j in range(i+1,len(param_names_plot)):
            ax = g.subplots[j,i]            # get mean:
            m1, m2 = chain_gauss.getMeans(pars=[chain_gauss.index[name]
                           for name in [KL_param_names_plot[i], KL_param_names_plot[j]]])
            #ax.scatter(np.exp(m1), np.exp(m2), color='k')
            alpha = 3.*np.linspace(-1.,1.,100)
            list_ind = [chain_gauss.index[name]
                           for name in [KL_param_names_plot[i], KL_param_names_plot[j]]]
            i1 = list_ind[0]
            j1 = list_ind[1]
            for k in range(nmod):

                ax.plot(np.exp(m1+alpha*param_directions[:,k][i1]), np.exp(m2+alpha*param_directions[:,k][j1]), color=sns.hls_palette(6)[k], label='Mode '+str(k+1))
    g.fig.legend(*ax.get_legend_handles_labels())






# In[ ]:
