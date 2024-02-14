import torch as th
import torchvision as tv
import generator
from torch.utils.data import TensorDataset

# select the device to be used for training
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# def npy_loader(path):
#   arr = np.load(path)['arr_0']
#   listt = np.array([np.reshape(x / 127.5 - 1, (3, 256, 256)) for x in arr[:1000]])
#   sample = th.from_numpy(listt).type(th.float32)
#   del listt
#   return sample


def setup_data(download=False):
    """
    setup the CIFAR-10 dataset for training the CNN
    :param batch_size: batch_size for sgd
    :param num_workers: num_readers for data reading
    :param download: Boolean for whether to download the data
    :return: classes, trainloader, testloader => training and testing data loaders
    """
    # data setup:
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    transforms = tv.transforms.ToTensor()

    trainset = tv.datasets.ImageFolder(root=path,
                                       transform=transforms)

    testset = tv.datasets.ImageFolder(root=path,
                                      transform=transforms)
    # trainset = tv.datasets.CIFAR10(root=data_path,
    #                                transform=transforms,
    #                                download=download)

    # testset = tv.datasets.CIFAR10(root=data_path,
    #                               transform=transforms, train=False,
    #                               download=False)
    # trainset = tv.datasets.DatasetFolder(
    #     root=data_path,
    #     loader=npy_loader,
    #     extensions='.npz'
    # )
    # trainset =  TensorDataset(npy_loader(data_path +"/sub/Train.npz"))
    # testset = None
    return classes, trainset, testset


if __name__ == '__main__':
    path = "../ZeldaALinkToThePast-Split"

    # some parameters:
    depth = 7
    START_DEPTH = 1
    START_EPOCH = 1
    # hyper-parameters per depth (resolution)
    num_epochs = [40, 70, 110, 160, 200, 250, 310, 400]
    fade_ins = [60, 60, 60, 60, 60, 60, 60, 60, 60]
    batch_sizes = depth * [9]
    latent_size = 512
    # get the data. Ignore the test data and their classes
    _, dataset, _ = setup_data(download=False)
    print(dataset)
    print([dataset[i][0].shape for i in range(1)])
    # dataset1 = np.load(path + '/Data/Train_5cm_norm.npz')
    # dataset2 = dataset1['arr_0']
    # print(dataset2.shape)
    # dataset = TensorDataset(*dataset2)
    # ======================================================================
    # This line creates the PRO-GAN
    # ======================================================================
    # pro_gan = ConditionalProGAN(num_classes=10, depth=depth,
    #                                latent_size=latent_size, device=device)
    pro_gan = generator.ProGAN(depth=depth,
                               latent_size=latent_size, device=device)
    # ======================================================================

    # ======================================================================
    # This line trains the PRO-GAN
    # ======================================================================
    model_version = "9"

    pro_gan.train(
        dataset=dataset,
        epochs=num_epochs,
        fade_in_percentage=fade_ins,
        batch_sizes=batch_sizes,
        log_dir="../models_"+model_version+"/",
        sample_dir="../samples_"+model_version+"/",
        save_dir="../models_"+model_version+"/",
        start_depth=START_DEPTH,
        start_epoch=START_EPOCH
    )
    # ======================================================================
