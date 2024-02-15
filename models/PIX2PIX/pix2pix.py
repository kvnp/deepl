from utils import DataGenerator, get_discriminator, \
    get_generator_unet, get_generator_training_model

from keras.optimizers import Adam
import matplotlib.pyplot as plt
from pandas import DataFrame
import tensorflow as tf
import numpy as np
import random
import os


class Pix2pix:
    def __init__(self, input_path, output_path, weight_path):
        self.image_id_list = os.listdir(os.path.join(input_path, "original"))
        self.train_path = input_path
        self.output_path = output_path

        seed = 2019
        np.random.seed = seed
        random.seed = seed
        tf.random.set_seed(seed)
        self.path = os.getcwd()

        self.image_size = (256, 256)
        train_path = "./dataset/"
        self.epochs = 300
        self.validation_size = 16

        self.train_gen = DataGenerator(
            self.image_id_list[:int(len(self.image_id_list)*0.8)],
            train_path,
            batch_size=2,
            image_size=self.image_size,
            channels=3
        )

        self.val_gen = DataGenerator(
            self.image_id_list[int(len(self.image_id_list)*0.8):],
            train_path,
            batch_size=2,
            image_size=self.image_size,
            channels=3
        )

        self.debug = False
        #image_source_dir = './dataset/facades/'
        self.direction = 'b2a'
        self.input_channel = 3  # input image channels
        self.output_channel = 3  # output image channels
        lr = 0.0002
        self.epoch = 1
        self.crop_from = 286
        #image_size = (640,480)
        batch_size = 2
        self.combined_filepath = os.path.join(weight_path, 'best_weights.h5')
        self.generator_filepath = os.path.join(weight_path, 'generator.h5')
        seed = 9584
        self.imagenet_mean = np.array([0.5, 0.5, 0.5])
        self.imagenet_std = np.array([0.5, 0.5, 0.5])

        self.train_step_per_epoch = int(len(self.image_id_list)*0.8) / batch_size + 1
        self.test_step_per_epoch = int(len(self.image_id_list)*0.2) / batch_size + 1
        self.train_image_generator =  DataGenerator(
            self.image_id_list[:int(len(self.image_id_list)*0.8)],
            train_path,
            batch_size=2,
            image_size=self.image_size,
            channels=3
        )

        self.test_image_generator = DataGenerator(
            self.image_id_list[int(len(self.image_id_list)*0.8):],
            train_path,
            batch_size=2,
            image_size=self.image_size,
            channels=3
        )

        summary = ""
        opt1 = Adam(learning_rate=lr)
        opt2 = Adam(learning_rate=lr)
        self.discriminator = get_discriminator()
        print(self.discriminator.summary())
        summary += self.discriminator.summary()
        summary += '\n'

        self.generator = get_generator_unet()
        self.generator.compile(optimizer=opt2, loss='mae', metrics=['mean_absolute_percentage_error', "mean_absolute_percentage_error"])
        print(self.generator.summary())
        summary += self.generator.summary()
        summary += '\n'

        self.generator_train = get_generator_training_model(self.generator, self.discriminator)
        print(self.generator_train.summary())
        summary += self.generator_train.summary()
        summary += '\n'

        with open(os.path.join(output_path, 'model_summary.txt'), 'w') as f:
            f.write(summary)
        del summary

        if os.path.exists(self.combined_filepath):
            self.generator_train.load_weights(self.combined_filepath, by_name=True)
            self.generator.load_weights(self.generator_filepath, by_name=True)
            print('weights loaded!')

        self.discriminator.compile(
            optimizer=opt1,
            loss='mse',
            metrics=['acc'],
            loss_weights=None
        )

        self.generator_train.compile(
            optimizer=opt2,
            loss=['mse', 'mae'],
            metrics=['mean_absolute_percentage_error', "acc"],
            loss_weights=[1, 10]
        )
        
        self.real = np.ones((batch_size, 16, 16, 1))
        self.fake = np.zeros((batch_size, 16, 16, 1))
        self.best_loss = 1000000

    def train(self):
        self.discriminator_dict = {
            'epoch': [],
            'train_step': [],
            'loss d_fake': [],
            'loss d_real': [],
            'fake_acc': [],
            'real_acc': []
        }

        self.generator_dict = {
            'epoch': [],
            'train_step': [],
            'loss fool': [],
            'loss g': []
        }
        try:
            for i in range(self.epochs):
                os.mkdir("./samples/{}".format(i))

                train_step = 0
                for imgA, imgB in self.train_image_generator:
                    train_step += 1
                    if train_step > self.train_step_per_epoch:
                        test_step = 0
                        total_loss = 0
                        total_mape = 0
                        for imgA, imgB in self.test_image_generator:
                            test_step += 1
                            if test_step > self.test_step_per_epoch:
                                break
                            gloss, mape = self.generator.test_on_batch(imgA, imgB)
                            total_loss += gloss
                            total_mape += mape

                        if total_loss / (test_step - 1) < best_loss:
                            print('test loss improved from {} to {}'.format(best_loss, total_loss / (test_step - 1)))
                            self.generator_train.save_weights(self.combined_filepath, overwrite=True)
                            self.generator.save_weights(self.generator_filepath, overwrite=True)
                            best_loss = total_loss / (test_step - 1)
                        break
                    self.discriminator.trainable = True
                    fakeB = self.generator.predict(imgA.reshape(2, *self.image_size, 3))

                    f, axes = plt.subplots(1, 3, figsize=(15, 10))
                    axes[0].imshow(imgB[0])
                    axes[1].imshow((fakeB[0] * 255).astype(np.uint8))
                    axes[2].imshow(imgA[0])
                    f.savefig(os.path.join(self.output_path, 'samples', f'{i}', f'{i}_{train_step}.png'))
                    plt.close(f)

                    loss_fake, fake_acc = self.discriminator.train_on_batch(np.concatenate((imgA, fakeB), axis=-1), fake)
                    loss_real, real_acc = self.discriminator.train_on_batch(np.concatenate((imgA, imgB), axis=-1), real)
                    if train_step % 20 == 0:
                        print('epoch:{} train step:{}, loss d_fake:{:.2}, loss d_real:{:.2}, fake_acc:{:.2}, real_acc:{:.2}'.format(i + 1, train_step, loss_fake, loss_real, fake_acc, real_acc))
                        self.discriminator_dict['train_step'].append(train_step * (i + 1))
                        self.discriminator_dict['loss d_fake'].append(loss_fake)
                        self.discriminator_dict['loss d_real'].append(loss_real)
                        self.discriminator_dict['fake_acc'].append(fake_acc)
                        self.discriminator_dict['real_acc'].append(real_acc)

                    self.discriminator.trainable = False
                    loss = self.generator_train.train_on_batch([imgA, imgB], [self.real, imgB])
                    if train_step % 20 == 0:
                        print('epoch:{} train step:{} loss fool:{:.2} loss g:{:.2}'.format(i + 1, train_step, loss[1], loss[0] - loss[1]))
                        self.generator_dict['train_step'].append(train_step * (i + 1))
                        self.generator_dict['loss fool'].append(loss[1])
                        self.generator_dict['loss g'].append(loss[0] - loss[1])
            
            self.save_to_output()
        except KeyboardInterrupt:
            print("Interrupt received, stopping training, saving data")
            self.save_to_output()

    def save_to_output(self):
        try:
            discriminator_csv = os.path.join(
                self.output_path,
                'train_discriminator.csv'
            )

            generator_csv = os.path.join(
                self.output_path,
                'train_generator.csv'
            )

            discriminator_df = DataFrame(self.discriminator_dict)
            generator_df = DataFrame(self.generator_dict)
            discriminator_df.to_csv(discriminator_csv)
            generator_df.to_csv(generator_csv)

            self.generator_train.save_weights(
                self.combined_filepath,
                overwrite=True
            )

            self.generator.save_weights(
                self.generator_filepath,
                overwrite=True
            )
        except KeyboardInterrupt:
            print("Interrupt received, stopping saving data")


    def _save_plots(self):
        plt.plot(self.discriminator_dict['train_step'], self.discriminator_dict['loss d_fake'], label='loss d_fake')
        plt.plot(self.discriminator_dict['train_step'], self.discriminator_dict['loss d_real'], label='loss d_real')
        plt.plot(self.discriminator_dict['train_step'], self.discriminator_dict['fake_acc'], label='fake_acc')
        plt.plot(self.discriminator_dict['train_step'], self.discriminator_dict['real_acc'], label='real_acc')
        plt.legend()
        plt.savefig(os.path.join(self.output_path, 'discriminator_loss.png'))
        plt.close()

        plt.plot(self.generator_dict['train_step'], self.generator_dict['loss fool'], label='loss fool')
        plt.plot(self.generator_dict['train_step'], self.generator_dict['loss g'], label='loss g')
        plt.legend()
        plt.savefig(os.path.join(self.output_path, 'generator_loss.png'))
        plt.close()

    def _save_samples(self, epoch, train_step, images):
        f, axes = plt.subplots(1, len(images), figsize=(15, 10))
        for i, img in enumerate(images):
            axes[i].imshow(img)
        
        #axes[0].imshow(images[0])
        #axes[1].imshow((fakeB[0] * 255).astype(np.uint8))
        path = os.path.join(self.output_path, 'samples', f'{epoch}')
        file = os.path.join(path, f'{epoch}_{train_step}.png')
        if not os.path.exists(path):
            os.mkdir(path)

        f.savefig(file)
        plt.close(f)