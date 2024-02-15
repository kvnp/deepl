import torch
import torchvision
import numpy as np
from torch.autograd import Variable
from models.SNGAN.SNGAN_Zelda import Generator

_class_definitions = ["ruins","waterfall",
                      "desert","village",
                      "woods","skyisland",
                      "mountains","central"]

def generate_img(output_dir, gen_dict, latent_size, num_classes,amount, device):
    gen = Generator(latent_size, num_classes).to(device)
    gen.load_state_dict(torch.load(gen_dict))
    
    img_batch = []
    labels = []
    for i in range(amount):
        labels_fake = Variable(torch.LongTensor(np.random.randint(0, num_classes, 1))).to(device)
        rand_X = torch.FloatTensor(np.random.randn(1, latent_size)).to(device)
    
        label = _class_definitions[int(labels_fake)]
        labels.append(label)
        print(f"class {i} is: {label}")      
        img_batch.append(gen(rand_X, labels_fake).detach())
    
    for i, img in enumerate(img_batch):
        torchvision.utils.save_image(img_batch[i], f"{output_dir}/{i:03}_{labels[i]}.png", normalize=True, value_range=(-1.0, 1.0))