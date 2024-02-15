import torch
import torchvision
import os
import pickle
import statistics
import numpy as np
from torch.autograd import grad as torch_grad
from torchvision import transforms
from tqdm import tqdm
from torch.autograd import Variable

from utils.evaluation import make_graph_sngan
import data.train.train_SNGAN.zelda_dataset as Z_Data
import models.SNGAN.SNGAN_Zelda as ZelNet

class trainer():
    def __init__(self,
                 output_dir,
                 n_epoch            = 15,
                 device             = "cpu",
                 check_point        = None,
                 latent_size        = 128,
                 class_num          = 8, 
                 save_state_every   = 5,
                 save_sample_every  = 3,
                 batch_size         = 16,
                 n_dis_update       = 5,
                 lr_g               = 0.001,
                 lr_d               = 0.0001,
                 gp_weight          = 10):
        
        self.output_dir = output_dir
        self.n_epoch = n_epoch
        self.device = torch.device(device)
        
        # checkpoint muss eine liste sein
        if check_point is not None:
            self.start_epoch = int(check_point[0]) if check_point is not None else 0
            self.state_gen = check_point[1]
            self.state_dis = check_point[2]
        else:
            self.start_epoch = 0
        
        self.latent_size = latent_size
        self.class_num = class_num
        self.save_state_every = save_state_every
        self.save_sample_every = save_sample_every
        self.batch_size = batch_size
        self.n_dis_update = n_dis_update
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.gp_weight = gp_weight
        
        self.dataloader = self._build_dataloader(img_dir= "data\\train\\train_SNGAN",
                                                label_path= "data\\train\\train_SNGAN\\labels.csv",
                                                batch_size=self.batch_size,
                                                classes= self.class_num)
        
        
    
    def train(self):
        # initialisieren der Models
        model_G = ZelNet.Generator(self.latent_size, n_classes_g=self.class_num)
        model_D = ZelNet.Discriminator(n_classes_d=self.class_num).to(self.device)
        model_G, model_D = model_G.to(self.device), model_D.to(self.device)
        
        # falls von state aus weitergemacht werden soll
        if self.start_epoch != 0:
            model_G.load_state_dict(torch.load(self.state_gen))
            model_D.load_state_dict(torch.load(self.state_dis))
            
        optim_G = torch.optim.Adam(model_G.parameters(), lr=self.lr_g, betas=(0.5, 0.999))
        optim_D = torch.optim.Adam(model_D.parameters(), lr=self.lr_d, betas=(0.5, 0.999))
        
        result = {"d_loss": [], "g_loss": [], "grad_pen": []}
        n = len(self.dataloader)
        
        
        #################################
        #           training            #
        #################################
        print("Starting training:")       
        for epoch in range(self.start_epoch, self.n_epoch):
            log_loss_D, log_loss_G, log_GP = [], [], []
            self.save_sample_every = self._increase_sample_rate(epoch, self.save_sample_every)
            
            
            for i, (real_img, labels_real) in tqdm(enumerate(self.dataloader), total= n):
                batch_len = len(real_img)
                real_img = real_img.to(self.device)
                
                
                # train D
                if i is not n-1:
                    # fake data
                    labels_fake = Variable(torch.LongTensor(np.random.randint(0, self.class_num, self.batch_size))).to(self.device)
                    rand_X = torch.FloatTensor(np.random.randn(batch_len, self.latent_size)).to(self.device)
                    fake_img = model_G(rand_X, labels_fake)
                    fake_img_tensor = fake_img.detach()
                
                    # train real
                    validity_real_img = model_D(real_img, labels_real)
                    
                    # train fake
                    validity_fake_img = model_D(fake_img_tensor, labels_fake)
                    
                    # grad penalty
                    grad_penalty = self._get_gradient_penalty(real_img, fake_img_tensor, labels_real, model_D, self.gp_weight)
                    log_GP.append(grad_penalty.item())
                    
                    # total loss
                    optim_D.zero_grad()
                    loss_d = validity_fake_img.mean() - validity_real_img.mean() + grad_penalty
                    
                    # backprop
                    loss_d.backward()
                    optim_D.step()
                    
                    # record dis loss
                    log_loss_D.append(loss_d.item())
                
                
                # train G
                # if i is not n-1:
                if i% self.n_dis_update == 0:
                    optim_G.zero_grad()

                    validity_gen = model_D(fake_img, labels_fake)
                    loss_g = -validity_gen.mean()

                    # backprop
                    loss_g.backward()
                    optim_G.step()

                    # record gen loss
                    log_loss_G.append(loss_g.item())
                    

            result["d_loss"].append(statistics.mean(log_loss_D))
            result["g_loss"].append(statistics.mean(log_loss_G))
            result["grad_pen"].append(statistics.mean(log_GP))
            print(f"epoch = {epoch}, \ng_loss = {result['g_loss'][-1]}, \nd_loss = {result['d_loss'][-1]} \ngrad_pen = {result['grad_pen'][-1]}")              
                
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
            if epoch % self.save_sample_every == 0:
                torchvision.utils.save_image(fake_img_tensor[:25], f"{self.output_dir}/epoch_{epoch:03}.png",
                                            nrow=5, padding=5, normalize=True, value_range=(-1.0, 1.0))


            if not os.path.exists(self.output_dir + "/models"):
                os.mkdir(self.output_dir+"/models")
            if epoch % self.save_state_every == 0:
                torch.save(model_G.state_dict(), f"{self.output_dir}/models/gen_epoch_{epoch:04}.pytorch")
                torch.save(model_D.state_dict(), f"{self.output_dir}/models/dis_epoch_{epoch:04}.pytorch")
            


        with open(self.output_dir + "/logs.pkl", "wb") as fp:
            pickle.dump(result, fp)
            
        make_graph_sngan(result_dir=self.output_dir, start_epoch=self.start_epoch)
                
    
    def _build_dataloader(self, img_dir, label_path, batch_size, classes, img_size = 171):
        
        img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(img_size, img_size),antialias=None),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip(p=0.5)
        ])
        label_transform = (lambda y: torch.zeros(classes, dtype=torch.int).scatter_(0, torch.tensor(y), value=1))
        dataset = Z_Data.Zelda_SNES_Map(label_path,img_dir,img_transform,label_transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def _increase_sample_rate(self, epoch, current_rate):
        if epoch >= 100:
            return 10
        elif epoch >= 500:
            return 20
        elif epoch >= 1000:
            return 100
        else:
            return current_rate

    # aus WGAN
    def _get_gradient_penalty(self, real_data, generated_data, labels, D, gp_weight= 10):
            batch_size = real_data.size()[0]

            # Calculate interpolation
            alpha = torch.rand(batch_size, 1, 1, 1)
            alpha = alpha.expand_as(real_data)
            alpha = alpha.to(self.device)
            interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
            interpolated = Variable(interpolated, requires_grad=True)
            interpolated = interpolated.to(self.device)

            # Calculate probability of interpolated examples
            prob_interpolated = D(interpolated, labels)

            # Calculate gradients of probabilities with respect to examples
            gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                                grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                                create_graph=True, retain_graph=True)[0]

            # Gradients have shape (batch_size, num_channels, img_width, img_height),
            # so flatten to easily take norm per example in batch
            gradients = gradients.view(batch_size, -1)
            # losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

            # Derivatives of the gradient close to 0 can cause problems because of
            # the square root, so manually calculate norm and add epsilon
            gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

            # Return gradient penalty
            return gp_weight * ((gradients_norm - 1) ** 2).mean()