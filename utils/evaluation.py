import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
import torchvision
from collections import OrderedDict
import pickle   
from models.SNGAN.SNGAN_Zelda import Generator
from torch.autograd import Variable
import torchvision.transforms.functional as F

label_dict = {"ruins": 0,
            "waterfall": 1,
            "desert": 2,
            "village": 3,
            "woods": 4,
            "sky_island": 5
        }

class_definitions = ["ruins","waterfall","desert","village","woods","sky_island"]

def testbilder():
    use_cuda = True
    batch_size = 50
    output_dir = "utils\\test"
    num_classes = 6
    latent_size = 128
    gen = Generator(latent_size, num_classes).to("cuda")
    gen.load_state_dict(torch.load("data\\pretrained\\SNGAN\\gen_epoch_0890.pytorch"))
    
    img_batch = []
    labels = []
    for i in range(batch_size):
        labels_fake = Variable(torch.LongTensor(np.random.randint(0, num_classes, 1))).to("cuda")
        rand_X = torch.FloatTensor(np.random.randn(1, latent_size)).to("cuda")
    
        label = class_definitions[int(labels_fake)]
        labels.append(label)
        print(f"class {i} is: {label}")      
        img_batch.append(gen(rand_X, labels_fake).detach())
    
    for i, img in enumerate(img_batch):
        torchvision.utils.save_image(img_batch[i], f"{output_dir}/{i:03}_{labels[i]}.png",
                                            nrow=10, padding=5, normalize=True, value_range=(-1.0, 1.0))