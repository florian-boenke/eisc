
import numpy as np 
import scipy as sc 
import scipy.optimize as opt
# from scipy.interpolate import interp  
import scipy.interpolate as interpolate 
import pandas as pd  
#from mat4py import loadmat 
import matplotlib.pyplot as plt 
import scipy.io 
import os 
from os import walk
import itertools 
#import astropy
#import astropy.coordinates  
import astropy.datetime 
#import astropy.units as au 

#%% useful 

def spliceNoMutate(myArray,indexToRemove):
    return np.slice(myArray, 0,indexToRemove).concat(np.slice(myArray, indexToRemove+1)) 


#%%  read in 

# files = [] 

N = int( (6361145 - 6336660) /60 ) 
# for i in range(N): 
#     files = files.concatenate('') 
    
#%% collecting data

path_15uhf60 = r'C:\Users\konst\Documents\UNI.IBK\Uni\M.Sc. Physik\2. Semester     TR\Space Physics\EISCAT\EISCAT_data\2023-03-15_beata_60@uhfa-orig/'
# path_15UHF60 = 
path_15uhf5 = r'C:\Users\konst\Documents\UNI.IBK\Uni\M.Sc. Physik\2. Semester     TR\Space Physics\EISCAT\EISCAT_data\2023-03-15_beata_5@uhfa/'
path_14uhf60 = r'C:\Users\konst\Documents\UNI.IBK\Uni\M.Sc. Physik\2. Semester     TR\Space Physics\EISCAT\EISCAT_data\2023-03-14_beata_60@uhfa-orig/' 
path_14uhf5 = r'C:\Users\konst\Documents\UNI.IBK\Uni\M.Sc. Physik\2. Semester     TR\Space Physics\EISCAT\EISCAT_data\2023-03-14_beata_5@uhfa/'

def collections(paths, order=True):  
    # w order = Truen: 4 paths of order 15_60, 15_5, 14_60, 14_5
    #path1, path2, path3, path4  = paths
    
    f_ = [[]]
    
    # for (dirpath, dirnames, filenames) in walk(path_15uhf60):
    for j in range(len(paths)): 
        for filenames in os.listdir(paths[j]): 
            if filenames.endswith('.mat'): 
                f_[j].extend([filenames]) 
        f_.extend([[]])
    return f_, order 



#%%  load / read in data

def read_in(path, collection, int_sec, day):
    # int_sec, day: int of 60 or 5 for  60s / 5s intervals, day int 14 or 15 

    os.chdir(path) 
    data_tot = scipy.io.loadmat(collection[0]) 
    data_tt = scipy.io.loadmat(collection[0]) 
    print('Z: ', np.shape(np.array(data_tot['r_param']))) 
    keys = list(data_tot.keys()) 
    
    for c, key in enumerate(data_tot.keys()): 
        #if '__version__' != key != '__header__': 
        if c>3:
            value = [data_tot[key]]  
            for k in range(1, len(collection)): 
                print('k: ', k)
                data_app = scipy.io.loadmat(collection[k])
                print('b_2: ', np.shape([data_app[key]]) )
                print('b_1: ', np.shape(value) )
                value = np.append(value, [data_app[key]], axis=0) 
                print('v_i: ', np.shape(value) ) 
            print('v: ', np.shape(value)) 
            data_tot.update({key: value}) 
            print('--------------------------')
            value = value[0]
            value = value[..., np.newaxis] 
            for k in range(1, len(collection)):  
                print('k: ', k)
                data_app = scipy.io.loadmat(collection[k])
                print('p_2: ', np.shape(data_app[key][..., np.newaxis])) 
                print('p_1: ', np.shape(value) )
                value = np.append(value, data_app[key][..., np.newaxis], axis=-1) 
                print('v_i: ', np.shape(value) ) 
            print('v: ', np.shape(value)) 
            data_tt.update({key: value}) 
    return data_tt, data_tot 


#%%  e dens  

def get_params(data_tt): 
    data_tt
    params =  data_tt['r_param'] 
    print(params.shape)
    print(params)
    
    N_e = params[:, 0, :]  
    h = data_tt['r_h'].reshape(42, 404)
    return params, N_e, h  


#  finding maxima in columns 
def time_and_splice_out_dens(collection, params, N_e, dday, n): 
    # n: int in [0, 3] number of brightest columns to slice out 
    argmaxcol = np.argmax(np.sum(N_e, axis=0))  
    argmaxcol_2 = np.argmax(np.delete(np.sum(N_e, axis=0), [argmaxcol]))
    argmaxcol_3 = np.argmax(np.delete(np.sum(N_e, axis=0), [argmaxcol, argmaxcol_2])) 
    if n==0: argscol = np.array([])
    elif n==1: argscol = np.array([argmaxcol])
    elif n==2: argscol = np.array([argmaxcol, argmaxcol_2])
    elif n==3: argscol = np.array([argmaxcol, argmaxcol_2, argmaxcol_3])
    else: print('WARNING: too many columns to slice out!')
    argscol[::-1].sort()   
    
    excl_col =  [collection[argscol[j]] for j in range(len(argscol))]
    excl_col_idx = [] 
    t_col = [] 
    for k, fname in enumerate(collection):  
        bname = os.path.basename(fname).replace(".mat", "") 
        if bname in excl_col: 
            np.append(excl_col_idx, [k]) 
        else: 
    #     np.append(t60, [day + datetime.timedelta(seconds=bname)])
            t_col.append(t_sec_to_ref(dday, bname))  
    ne = np.zeros((len(N_e), len(N_e[-1])-len(excl_col))) 
    for i in range(len(N_e)): 
        for j in range(len(argscol)): 
            ne[i] = np.delete(N_e[i], argscol[j]) 
    N_e = ne 
    if N_e.shape[-1] != np.shape(t_col): print(f'WARNING: N_e and t_col not of same length\nwidth(N_e) = {N_e.shape[-1]}, len(t_col)= {len(t_col)}')  
    return N_e, t_col

def time_and_set_maxcols_avg(collection, params, N_e, dday, n):
    # n: int in [0, 3] number of brightest columns to slice out 
    argmaxcol = np.argmax(np.sum(N_e, axis=0))  
    argmaxcol_2 = np.argmax(np.delete(np.sum(N_e, axis=0), [argmaxcol]))
    argmaxcol_3 = np.argmax(np.delete(np.sum(N_e, axis=0), [argmaxcol, argmaxcol_2])) 
    if n==0: argscol = np.array([])
    elif n==1: argscol = np.array([argmaxcol])
    elif n==2: argscol = np.array([argmaxcol, argmaxcol_2])
    elif n==3: argscol = np.array([argmaxcol, argmaxcol_2, argmaxcol_3])
    else: print('WARNING: too many columns to slice out!')
    argscol[::-1].sort() 
    t_col = [] 
    # calc. average N_e over field to set the error column visually to it 
    avg = np.average(N_e) 
    for k, fname in enumerate(collection):  
        bname = os.path.basename(fname).replace(".mat", "")    
        t_col.append(t_sec_to_ref(dday, bname))  
        if k in argscol: 
            for i in range(len(N_e)): N_e[i, k] = 0 
    return N_e, t_col

#%% 

def t_sec_to_ref(dday, secs): 
    return dday + datetime.timedelta(secs) 

#def get_zenith():
#       
#    return 
    

#%%  interpolate & plot

#interpolation
def interpol(N_e, h, z_min, z_max, z2_min, z2_max, h_ints): 
    z = np.linspace(z_min, z_max, h_ints) 
    z2 = np.linspace(z2_min, z2_max, h_ints)
    
    Ne = np.zeros((len(z), len(N_e[-1]))) 
    Ne2 = np.zeros((len(z2), len(N_e[-1])))
    for i in range(N_e.shape[-1]): 
        Ne[:, i] = np.interp(z, h[:, i], N_e[:, i])
        Ne2[:, i] = np.interp(z2, h[:, i], N_e[:, i]) 
    return Ne, Ne2, z, z2

def plot_edens(order_int, Ne, z, t_col):
    orderstr = ['15uhf60, 15uhf5, 14uhf60, 14uhf5'] 
    secarr = [60, 5, 60, 5]
    fig, axs = plt.subplots(1, 1) 
    pmesh = axs.pcolormesh(t_col, z, Ne, cmap='plasma') 
    axs.set(xlabel = 'daytime', ylabel = '$z$ in km')
    axs.set_title(f'electron density\nUHF{dayarr[order_int]}.03.23, {secarr[order_int]}s') 
    #add colormap legend  
    box = axs.get_position() 
    width, pad = 0.05, 0.03
    cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height ]) 
    fig.colorbar(pmesh, cax=cax, label = r'$N_e$ in m${}^{-3}$') 
    #fig.tight_layout() 
    fig.savefig(f'images/plot_Ne_{orderstr[order_int]}_1.pdf') 
    return fig.show() 

#%% run   --  task 1

orderstr = ['15uhf60, 15uhf5, 14uhf60, 14uhf5'] 
secarr = [60, 5, 60, 5]
dayarr= [15, 15, 14, 14] 

paths = np.array([path_15uhf60, path_15uhf5, path_14uhf60, path_14uhf5])  
f_tt, order = collections(paths, order=True)

# for j in range(0, np.shape(paths)[0]): 
#     print('JJJJJJJJJJJJJJJ: ', j)
#     data_tt, data_tot = read_in(paths[j], f_tt[j], secarr[j], dayarr[j]) 
#     params, N_e, h = get_params(data_tt)
#     N_e, t_col = time_and_set_maxcols_avg(f_tt[j], params, N_e, 1) 
#     Ne, Ne2, z, z2 = interpol(N_e, h, 80, 600, 100, 300, 30)  
#     print('JJJJJJJJJJJJJJJ: ', j)
#     plot_edens(j, Ne, z, t_col) 
#     plot_edens(j, Ne2, z2, t_col) 

# functional calls: 
    
def coll_eval(path, f_col, sec, day, dday): 
    data_tt, data_tot = read_in(path, f_col, sec, day) 
    params, N_e, h = get_params(data_tt)
    N_e, t_col = time_and_set_maxcols_avg(f_col, params, N_e, dday, 1) 
    Ne, Ne2, z, z2 = interpol(N_e, h, 80, 600, 100, 300, 30)  
    return Ne, Ne2, z, z2, t_col

def plot1(order_int, Ne, Ne2, z, z2, t_col):
    plot_edens(order_int, Ne, z, t_col)
    plot_edens(order_int, Ne2, z2, t_col) 
    
def run1(j, path, ftt, sec, day, dday): 
    return plot1(j, *coll_eval(path, ftt, sec, day, dday))

def task1(paths, order = True, dday = datetime.datetime(2023,1,1)):
    if order == True: f_tt, order = collections(paths, order=True)
    for j in range(len(paths)): 
        run1(j, paths[j], f_tt[j], secarr[j], dayarr[j], dday) 

def init1(order_int, dday= datetime.datetime(2023, 1, 1)):
    f_tt, order = collections(paths, order=True)
    return coll_eval(paths[order_int], f_tt[order_int], secarr[order_int], dayarr[order_int], dday)  


#%%
task1(paths, True) 

#%% 2 cells for to call the plots seperately after loading all the data  --  load
# choose input j out of {0, 1, 2, 3}
j = 2
Ne, Ne2, z, z2, t_col = init1(j)

#%%  --  plot 

plot1(j, Ne, Ne2, z, z2, t_col) 

