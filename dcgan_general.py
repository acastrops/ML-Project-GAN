'''
Code is modified from tutorial by Felix Mohr
source: https://github.com/FelixMohr/Deep-learning-with-Python/blob/master/DCGAN-face-creation.ipynb
'''
import urllib.request
import tarfile
import os
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.misc import imresize, imsave
import tensorflow as tf
import glob
import shutil
plt.switch_backend('agg') # To not open window with plots on the server

# Path to directory containing all the training imgs
input_dir = './happy_kaggle_faces/'
input_path = os.path.join(input_dir,'*g') # will work for png or jpg

# All images that match input_path
input_files = glob.glob(input_path)

# Total number of input images
total_input = len(input_files)

# Read in one img to get the dimensions
example_input = imread(input_files[0])
w, h, c = example_input.shape # c = # of channels

# Create directory for output
out_dir = "kaggle_out_run1"
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)


# Method to pull random batch of images
def next_batch(num, data=input_files):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [imread(data[i]) for i in idx]

    shuffled = np.asarray(data_shuffle)

    return shuffled

# Code by Parag Mital (https://github.com/pkmital/CADL/)
# Makes montage of output images from the generator
def montage(images):
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
    elif len(images.shape) == 4 and images.shape[3] == 1:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 1)) * 0.5
    elif len(images.shape) == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1)) * 0.5
    else:
        raise ValueError('Could not parse image shape of {}'.format(
            images.shape))
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    return m

# Definition of the network
tf.reset_default_graph()
batch_size = 64 # number of real images fed in
n_noise = 64 # inital size of noise images

# Placeholder for real (X_in)
X_in = tf.placeholder(dtype=tf.float32, shape=[None, w, h, c], name='X')
# Placeholder for fake imgs (noise): feed a flat vector of len(n_noise) into generator
noise = tf.placeholder(dtype=tf.float32, shape=[None, n_noise])
# Proportion of neurons to keep after a dropout layer
keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
# Flag for training required for batch-norm layers
is_training = tf.placeholder(dtype=tf.bool, name='is_training')


def lrelu(x): # Leaky-relu to avoid the dying relu problem
    return tf.maximum(x, tf.multiply(x, 0.2))


def binary_cross_entropy(x, z): # For multi-label classifications where m=2; x: y, z: y-hat
    eps = 1e-12
    return (-(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps)))

# img_in: list of image arrays,
# There are two instances of the discriminator (1 for real, 1 for fake)
# for the 1st instance (real), reuse=None
# for the 2nd instance (fake), reuse=True insures both instances are using the same weights
def discriminator(img_in, reuse=None, keep_prob=keep_prob):
    activation = lrelu
    with tf.variable_scope("discriminator", reuse=reuse):
        #
        x = tf.reshape(img_in, shape=[-1, w, h, c]) # -1: invariate to batch size
        #
        x = tf.layers.conv2d(x, kernel_size=5, filters=256, strides=2,
                            padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)

        x = tf.layers.conv2d(x, kernel_size=5, filters=128, strides=1,
                            padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)

        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1,
                            padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=128, activation=activation)
        x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
        return x

def generator(z, keep_prob=keep_prob, is_training=is_training):
    # the factor required to get the noise to the right size, ORIGINAL IMGS MUST HAVE DIM DIVISIBLE BY 4 (unless we change the strides)
    d = 4

    # the noise fed into the generator is smaller than the input imgs and grows to the input size as it flows through the generator
    noise_w = int(w/d)
    noise_h = int(h/d)

    # custom activation function to use in each layer except the last
    activation = lrelu
    momentum = 0.9
    with tf.variable_scope("generator", reuse=None):
        x = z
        #
        # d1 = 4#3
        # d2 = c

        x = tf.layers.dense(x, units=d * d * c, activation=activation)

        x = tf.layers.dropout(x, keep_prob)

        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)

        x = tf.reshape(x, shape=[-1, d, d, c])

        x = tf.image.resize_images(x, size=[noise_w, noise_h])


        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=256, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)

        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=128, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)

        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)

        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=c, strides=1, padding='same', activation=tf.nn.sigmoid)

        return x

# loss function and optimizers
g = generator(noise, keep_prob, is_training)
print(g)
d_real = discriminator(X_in)
d_fake = discriminator(g, reuse=True)

vars_g = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]


d_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_d)
g_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_g)

loss_d_real = binary_cross_entropy(tf.ones_like(d_real), d_real)
loss_d_fake = binary_cross_entropy(tf.zeros_like(d_fake), d_fake)
loss_g = tf.reduce_mean(binary_cross_entropy(tf.ones_like(d_fake), d_fake))

loss_d = tf.reduce_mean(0.5 * (loss_d_real + loss_d_fake))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(loss_d + d_reg, var_list=vars_d)
    optimizer_g = tf.train.RMSPropOptimizer(learning_rate=0.0002).minimize(loss_g + g_reg, var_list=vars_g)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# training
print("training")
num_iterations = 10000
for i in range(num_iterations):
    train_d = True
    train_g = True
    keep_prob_train = 0.6 # 0.5


    n = np.random.uniform(0.0, 1.0, [batch_size, n_noise]).astype(np.float32)
    batch = [b for b in next_batch(num=batch_size)]

    d_real_ls, d_fake_ls, g_ls, d_ls = sess.run([loss_d_real, loss_d_fake, loss_g, loss_d], feed_dict={X_in: batch, noise: n, keep_prob: keep_prob_train, is_training:True})

    d_fake_ls_init = d_fake_ls

    d_real_ls = np.mean(d_real_ls)
    d_fake_ls = np.mean(d_fake_ls)
    g_ls = g_ls
    d_ls = d_ls

    if g_ls * 1.35 < d_ls:
        train_g = False
        pass
    if d_ls * 1.35 < g_ls:
        train_d = False
        pass

    if train_d:
        sess.run(optimizer_d, feed_dict={noise: n, X_in: batch, keep_prob: keep_prob_train, is_training:True})


    if train_g:
        sess.run(optimizer_g, feed_dict={noise: n, keep_prob: keep_prob_train, is_training:True})


    if not i % 10:
        print('Iter: {}'.format(i))
        print('D loss: {:.4}'.format(d_ls))
        print('G_loss: {:.4}'.format(g_ls))
        print()
        if not train_g:
            print("not training generator")
        if not train_d:
            print("not training discriminator")
        gen_imgs = sess.run(g, feed_dict = {noise: n, keep_prob: 1.0, is_training:False})
        imgs = [img[:,:,:] for img in gen_imgs]
        m = montage(imgs[0:16])
        #m = imgs[0]
        plt.axis('off')
        plt.imshow(m, cmap='gray')
        plt.savefig('{0}/{1}.png'.format(out_dir, str(i).zfill(5)), bbox_inches='tight')
