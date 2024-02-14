import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
import pickle



def look_at_samplebatch(img_batch, use_cuda= False):
    counter = 0
    if use_cuda:
        img_batch = img_batch.cpu()
    for raw_image in img_batch:
        raw_image = raw_image.detach().numpy()
        print("########Bild",counter,": ", raw_image.shape)
        # print("imdata: ", test)
        cv2_image = np.transpose(raw_image, (1, 2, 0))
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        cv2.imshow("bilder",cv2_image)
        cv2.waitKey(0)
    quit()
    
    
def remove_prefix(state_dict):
    new_dict = OrderedDict()
    
    for key, value in state_dict.items():
        name = key[7:] # remove `module.`
        new_dict[name] = value
    
    return new_dict
    
def make_graph_sngan():
    with open('data\\results\\results_11\\logs_1.pkl', 'rb') as f:
        stats = pickle.load(f)
    
    with open('data\\results\\results_11\\logs.pkl', 'rb') as f:
        stats_1 = pickle.load(f)
    
    
    start_epoch = 200
    
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
    
def make_graph_warp_space():
    stats = json.load(open("experiments\\complete\\ProgGAN-ResNet-K128-D32-LearnGammas-eps0.1_0.2\\stats.json", 'r'))
    
    x_epoch         = np.arange(10, (len(stats)+1)*10, 10)
    y_accuracy      = np.zeros([len(stats)])
    y_classf_loss   = np.zeros([len(stats)])
    y_reg_loss      = np.zeros([len(stats)])
    y_total_loss    = np.zeros([len(stats)])
    
    for i in range(len(stats)):
        data = stats[str((i+1)*10)]
        y_accuracy[i]       = data["accuracy"]
        y_classf_loss[i]    = data["classification_loss"]
        y_reg_loss[i]       = data["regression_loss"]
        y_total_loss[i]     = data["total_loss"]
        
    fig, axes = plt.subplots(4, figsize=(7,6))
    fig.suptitle("Training Statistics")
    axes[0].plot(x_epoch[0:1000:50],y_accuracy[0:1000:50], color='green')
    axes[1].plot(x_epoch[0:1000:50],y_classf_loss[0:1000:50], color='orange')
    axes[2].plot(x_epoch[0:1000:50],y_reg_loss[0:1000:50], color='orange')
    axes[3].plot(x_epoch[0:1000:50],y_total_loss[0:1000:50], color='maroon')
    
    plt.xlabel('epoch')
    axes[0].set_ylabel("accuracy")
    axes[1].set_ylabel("classf_loss")
    axes[2].set_ylabel("reg_loss")
    axes[3].set_ylabel("total_loss")
    
    for i in range(4):
        axes[i].grid(True)
        axes[i].set_xlim(xmin=0)
        # axes[i].set_ylim(ymin=0)
    
    plt.subplots_adjust(hspace= 1)
    plt.show()
        
make_graph_sngan()