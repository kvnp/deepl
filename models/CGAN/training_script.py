import torch as th
import torchvision as tv
import models.CGAN.generator as gen

# select the device to be used for training
device = th.device("cuda" if th.cuda.is_available() else "cpu")


def setup_data(path):
    transforms = tv.transforms.ToTensor()
    return tv.datasets.ImageFolder(root=path, transform=transforms)


def training(num_epochs, fade_ins, output_path="./sample/", dataset_path="./ZeldaALinkToThePast-Split"):
    depth = 7
    start_depth = 1
    start_epoch = 1

    # hyper-parameters per depth (resolution)
    if not num_epochs:
        num_epochs = [70, 110, 160, 200, 250, 310, 400]
    if not fade_ins:
        fade_ins = [50, 50, 50, 50, 50, 50, 50, 50]
    batch_sizes = depth * [7]
    latent_size = 512

    # ======================================================================
    # load and output the dataset format
    dataset = setup_data(dataset_path)
    print("\nDataset was initialized")
    print(dataset)
    print([dataset[i][0].shape for i in range(1)])
    print("Generator is being initialized")
    # ======================================================================
    pro_gan = gen.ProGAN(depth=depth)
    print("Initialization complete\n")
    # ======================================================================
    pro_gan.train(
        dataset=dataset,
        epochs=num_epochs,
        fade_in_percentage=fade_ins,
        batch_sizes=batch_sizes,
        log_dir=output_path+"_logs",
        sample_dir=output_path+"_samples",
        save_dir=output_path+"_model",
        start_depth=start_depth,
        start_epoch=start_epoch
    )
    # ======================================================================
