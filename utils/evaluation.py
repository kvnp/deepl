import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from collections import OrderedDict
import pickle   
from models.SNGAN.SNGAN_Zelda import Generator
from torch.autograd import Variable
import torchvision.transforms.functional as F

def make_graph_sngan(result_dir, start_epoch = 0):
    with open(result_dir+"\\logs.pkl", 'rb') as f:
        stats = pickle.load(f)
    
    g_loss  = np.array(stats["g_loss"])
    d_loss  = np.array(stats["d_loss"])
    gr_pen  = np.array(stats["grad_pen"])
    x_epoch = np.arange(start_epoch, start_epoch+len(g_loss))
    
    g_line = plt.plot(x_epoch, g_loss, color="red")
    d_line = plt.plot(x_epoch, d_loss, color="blue")
    gp_line= plt.plot(x_epoch, gr_pen, color="green")
    
    plt.xlim(xmin= 0)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True)
    
    plt.legend(["Loss Generator", "Loss Discriminator", "Gradient Penalty"])
    
    plt.show()
    
    g_loss  = np.array(stats["g_loss"])
    d_loss  = np.array(stats["d_loss"])
    gr_pen  = np.array(stats["grad_pen"])
    x_epoch = np.arange(start_epoch, start_epoch+len(g_loss))    
    
    g_line = plt.plot(x_epoch, g_loss, color="red")
    d_line = plt.plot(x_epoch, d_loss, color="blue")
    gp_line= plt.plot(x_epoch, gr_pen, color="green")
    
    plt.xlim(xmin= 0)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True)
    
    plt.legend(["Loss Generator", "Loss Discriminator", "Gradient Penalty"])
    
    plt.savefig(result_dir+"\\traing_statistics.png")

