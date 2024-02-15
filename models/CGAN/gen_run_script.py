import torch
import torchvision.utils as vutils
import os
import models.CGAN.generator as gen
import copy
from torch.nn.functional import interpolate


class ImageGenerator:
    def __init__(self, generator_path, depth, latent_size, output_dir, device=torch.device("cpu"), use_ema=True):
        """
        Initialize the ImageGenerator.
        """
        self.generator_path = generator_path
        self.depth = depth
        self.latent_size = latent_size
        self.output_dir = output_dir
        self.device = device
        self.use_ema = use_ema
        self.generator, self.gen_shadow = self.load_generator()

    def load_generator(self):
        """
        Load the pre-trained generator model.
        """
        pro_gan = gen.ProGAN(
            depth=self.depth,
            latent_size=self.latent_size,
            device=self.device,
            use_eql=True,
            use_ema=self.use_ema
        )
        pro_gan.gen.load_state_dict(torch.load(self.generator_path, map_location=self.device))
        gen_shadow = None
        if self.use_ema:
            gen_shadow = copy.deepcopy(pro_gan.gen)
        return pro_gan.gen, gen_shadow

    def generate_and_save_images(self, num_images=16, scale_factor=1):
        """
        Generate and save images using the pre-trained generator.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        with torch.no_grad():
            noise = torch.randn(num_images, self.latent_size, device=self.device)
            generator = self.gen_shadow if self.gen_shadow is not None else self.generator
            images = generator(noise, depth=self.depth-1, alpha=1)

            if scale_factor > 1:
                images = interpolate(images, scale_factor=scale_factor)

            for i, image in enumerate(images):
                vutils.save_image(image, os.path.join(self.output_dir, f"generated_image_{i+1}.png"), normalize=True)


def generate(num_pics=50, gen_path="../../data/pretrained/ProGAN/GAN_GEN_6.pth", output_dir="./pictures_outputs"):
    depth = 7
    latent_size = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Generator initialization")
    image_generator = ImageGenerator(gen_path, depth, latent_size, output_dir, device, use_ema=True)
    print("Generator ready, pictures generation")
    image_generator.generate_and_save_images(num_images=int(num_pics), scale_factor=1)
    print("Pictures generation complete!")
