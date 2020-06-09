
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from keras.preprocessing import image
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D,Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras import layers
import tensorflow as tf
from keras.initializers import RandomNormal

from keras import backend

# implementation of wasserstein loss
from keras.constraints import Constraint
from keras import backend
kernal=3
from PIL import Image
# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='GPU')

# To find out which devices your operations and tensors are assigned to


noise_size=100
img_size=128
reshape_size=img_size*img_size*3

def preprocess_data():
    arr = []
    path = "C:/Users/seb/PycharmProjects/nart/flow"
    dirs = os.listdir(path)
    for item in dirs:
        if (item == '.DS_Store'):
            continue
        X_test = image.load_img('flow/' + item, target_size=(img_size, img_size),
                                color_mode="rgb");  # loading image and then convert it into grayscale and with it's target size
        X_test = image.img_to_array(X_test);  # convert image into array
        arr.append(X_test)
    return arr
# def generate_imgnoise(a):
#     img = (np.random.standard_normal([a,200, 200, 3]) * 255).astype(np.uint8)
#     return img
def generate_imgnoise(a):
    noise = np.random.normal(0, 1, (a, noise_size))
    return noise

#
#
# def build_generator():
#     model = Sequential()
#     model.add(Dense(256,input_shape=(noise_size,)))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(BatchNormalization(momentum=0.8))
#
#     model.add(Dense(512))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(BatchNormalization(momentum=0.8))
#
#     model.add(Dense(512))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(BatchNormalization(momentum=0.8))
#
#
#     model.add(Dense(reshape_size, activation='tanh'))
#     model.add(Reshape((img_size,img_size,3)))
#     model.summary()
#     noise = Input(shape=(noise_size,))
#     img = model(noise)
#     mod=Model(noise,img)
#     mod.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5))
#     return mod
# #
#
# def build_generator():
#     init = RandomNormal(stddev=0.02)
#     # define model
#     model = Sequential()
#     # foundation for 7x7 image
#     input_shape=4
#     input_size=(input_shape**2)*256
#
#     model.add(Dense(input_size, kernel_initializer=init, input_dim=(100)))
#     model.add(Reshape((input_shape, input_shape, 256)))
#     model.add(LeakyReLU(alpha=0.2))
#
#     # upsample to 8*8
#     model.add(Conv2DTranspose(512, (4,4), strides=(2, 2), padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))
#     # upsample to 16*16
#     model.add(Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))
#     # output 32*32
#     model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))
#
#     model.add(Conv2D(3, (3, 3), strides=(1,1),activation='tanh', padding='same'))
#     img=Input(shape=(100,))
#     d=model(img)
#     mod=Model(img,d)
#     return mod
# def build_generator():
#     input_shape=8
#     input_size=(input_shape**2)*256
#
#     model=Sequential()
#     model.add(Dense(128, input_shape=(100,)))
#     model.add(Dense(256))
#     model.add(Dense(512))
#     model.add(Dense(512))
#     model.add(Dense(input_size))
#     model.add(Reshape((input_shape, input_shape, 256)))
#
#     model.add(Conv2DTranspose(512, kernal, strides=(2, 2), padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Conv2DTranspose(256, kernal, strides=(2, 2), padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Conv2DTranspose(128, kernal, strides=(2, 2), padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))
#
#     model.add(Conv2D(3, (7 ,7), strides=(1,1),activation='tanh', padding='same'))
#
#     img=Input(shape=(100,))
#     d=model(img)
#     mod=Model(img,d)
#     return mod

#
# def build_discrim():
#     model=Sequential()
#     model.add(Flatten(input_shape=(img_size,img_size,3)))
#     model.add(Dense(512))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Dense(256))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Dense(128))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Dense(1, activation='sigmoid'))
#     model.summary()
#     img=Input(shape=(img_size,img_size,3))
#     d=model(img)
#     mod=Model(img,d)
#     return mod
# class ClipConstraint(Constraint):
#     # set clip value when initialized
#     def __init__(self, clip_value):
#         self.clip_value = clip_value
#
#     # clip model weights to hypercube
#     def __call__(self, weights):
#         return backend.clip(weights, -self.clip_value, self.clip_value)
#
#     # get the config
#     def get_config(self):
#         return {'clip_value': self.clip_value}
def build_discrim():
    ran=RandomNormal(stddev=0.02)
    # clip=ClipConstraint(0.01)

    model=Sequential()
    model.add(Conv2D(64, kernel_size=3, strides=(2, 2), padding='same', kernel_initializer=ran, input_shape=(img_size,img_size,3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, kernel_size=3, strides=(2, 2), padding='same', kernel_initializer=ran ))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, kernel_size=3, strides=(1 ,1), padding='same',  kernel_initializer=ran))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))
    model.summary()

    img=Input(shape=(img_size,img_size,3))
    d=model(img)
    mod=Model(img,d)
    return mod
def build_generator():
    model=Sequential()
    input_shape = 8
    input_size = (input_shape ** 2) * 256


    model.add(Dense(input_size, input_shape=(100,)))

    model.add(Reshape((input_shape, input_shape, 256)))
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3,  padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3,  padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3,  padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(UpSampling2D())

    # model.add(Conv2D(256, kernel_size=4,  padding='same'))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(3, kernel_size=3,  padding='same', activation=('tanh')))
    model.summary()


    img=Input(shape=(100,))
    d=model(img)
    mod=Model(img,d)
    return mod
def save_img(epoch,gen):

    noise = np.random.normal(0, 1, (1, noise_size))
    gen_img = gen.predict(noise)
    try:
        arr = (gen_img[0] * .5) + .5
        plt.imsave(('C:/Users/seb/PycharmProjects/nart/epochs_img/epoch-'+str(epoch)+'.png'), arr)
    except:
        arr = (gen_img[0]-min(gen_img[0]))/(max(gen_img[0])-min(gen_img[0]))
        plt.imsave(('C:/Users/seb/PycharmProjects/nart/epochs_img/epoch-'+str(epoch)+'.png'), arr)


def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)




def train(epochs, batch_size=128, save_interval=25):
    dis = build_discrim()
    gen = build_generator()
    z = Input(shape=(100,))
    img = gen(z)
    dis.compile(loss=wasserstein_loss, optimizer=RMSprop(lr=0.00005), metrics=['accuracy'])
    dis.trainable = False

    valid = dis(img)
    gan = Model(z, valid)
    gan.compile(loss=wasserstein_loss, optimizer=RMSprop(lr=0.00005), metrics=['accuracy'])



    data=np.array(preprocess_data())
    X_train = (data.astype(np.float32) - 127.5) / 127.5
    half=int(batch_size/2)
    training_dis=[]
    training_gen=[]

    print(epochs, "(epochs) : ",batch_size,"(batchsize)")
    for epoch in range(epochs):

    # train the discriminator
        for i in range(5):
            idx = np.random.randint(0, data.shape[0], half)
            real_im = X_train[idx]

            loss_dis_real=dis.train_on_batch(real_im,(-np.ones(half)))

            fake_im=gen.predict(generate_imgnoise(half))
            loss_dis_fake = dis.train_on_batch(fake_im, np.ones(half))
            # Clip critic weights
            for l in dis.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -0.01, 0.01) for w in weights]
                l.set_weights(weights)

        #train the generator

        noise=generate_imgnoise(half)
        g_loss=gan.train_on_batch(noise, -np.ones(half))

        d_loss = 0.5 * np.add(loss_dis_real, loss_dis_fake)
        training_gen.append(g_loss)
        training_dis.append(d_loss)
        print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss[0]))


        if epoch % save_interval == 0:
            save_img(epoch,gen)
    plt.plot(training_gen)
    plt.plot(training_dis)
    plt.show()


# feature testing
# valid_y = np.array([1] * 20)
# print(valid_y)
# data = np.array(preprocess_data())
# half = int(128 / 2)
# idx = np.random.randint(0, data.shape[0], half)
# print(data.shape)
#
#
# a=build_generator()
# b=build_discrim()
# noise = np.random.normal(0, 1, (1, 1000))
#
#
# # noise=generate_imgnoise(10)
# gen_img=a.predict(noise)
# arr=(gen_img[0]*.5)+.5
# plt.imsave('newimg.png',arr)
#
# plt.show()
# # plt.imshow(noise[2])
# # # plt.show()
# # print(noise[0].shape)
# from tensorflow.python.client import device_lib
# print(devic
# e_lib.list_local_devices())

train(batch_size=128,epochs=50000)
