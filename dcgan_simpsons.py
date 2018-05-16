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
# from scipy.misc import imresize, imsave
import tensorflow as tf
import glob
import shutil
plt.switch_backend('agg') # To not open window with plots on the server

# Path to directory containing all the training imgs
input_dir = './homer_simpson_60/'
input_path = os.path.join(input_dir,'*g') # will work for png or jpg

# All images that match input_path
input_files = glob.glob(input_path)

# Total number of input images
total_input = len(input_files)

# Read in one img to get the dimensions
example_input = imread(input_files[0])
w, h, c = example_input.shape # c = # of channels

# Create directory for output
out_dir = "homer_out_5"
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

# Method to pull the best of the generator's batch based on the discriminator scores.
def best_gen_imgs(gen_imgs, discrim_scores, num_best):
    best_indexes = np.flip(np.argsort(discrim_scores),0)[0:num_best]
    best_imgs = gen_imgs[best_indexes]
    return(best_imgs)

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

        # Input layer: takes list of image arrays and reshapes them into a tensor object
        # Shape: [batch_size, width, height, channels]: sizes for the 4 separate dimensions
        x = tf.reshape(img_in, shape=[-1, w, h, c]) # -1: invariate to batch size

        # First convolutional layer
        # Output shape: [batch_size, w/2, h/2, channels=filters]
        x = tf.layers.conv2d(x, kernel_size=5, filters=256, strides=2,
                            padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob) # sets a fraction (1-keep_prob) of the inputs to 0

        # Second convolutional layer
        # Output shape: [batch_size, w/2, h/2, channels=filters]
        x = tf.layers.conv2d(x, kernel_size=5, filters=128, strides=1,
                            padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)

        # Third convolutional layer
        # Output layer: [batch_size, w/2, h/2, channels=filters]
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1,
                            padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)

        # Reshaping output from 3rd conv. layer before sending it to the fully connected layer
        # Shape: [batch_size, w/2*h/2*channels=filters from 3rd convolution]
        x = tf.contrib.layers.flatten(x)
        # Reduces the dimensionality of the output space before lrelu
        x = tf.layers.dense(x, units=128, activation=activation) # Outputs [batch_size, units]
        # Gives the probability of the image being real; Outputs [batch_size, units]
        x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)

        # Returns a tensor [batch_size, 1]: a prob for each image in the batch
        return x

def generator(z, keep_prob=keep_prob, is_training=is_training):
    factor = 4 # Factor required to get the noise to the right size
               # ORIGINAL IMGS MUST HAVE DIM DIVISIBLE BY 4 (unless we change the strides)

    noise_w = int(w/factor) # Noise fed into the generator is smaller than the input images
    noise_h = int(h/factor) # and grows to the input size as it flows through the generator

    activation = lrelu # leaky-relu
    momentum = 0.9 # Used in batch_norm: decay for the moving average
    with tf.variable_scope("generator", reuse=None):
        # Input Layer (Noise)
        x = z # Shape: [batch_size, n_noise]

        # First layer (fully connected): reducing the noise vector by rescaling with 'factor'
        # Shape: here, we went from n_noise=64 to (factor**2)*c=48
        x = tf.layers.dense(x, units=factor * factor * 512, activation=activation)
        x = tf.layers.dropout(x, keep_prob)

        # First Batch Normalization layer:
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        # Reshape the noise vector into a stack of c matrices of size (factor x factor)
        x = tf.reshape(x, shape=[-1, factor, factor, 512])
        # Enlarge the noise images using binlinear interpolation
        # Resulting sizes: noise_w = original_w/factor noise_h = original_h/factor
        x = tf.image.resize_images(x, size=[noise_w, noise_h])

        # First convolutional layer: applies 256 different kernels over each image in the batch, with stride of 2
        # Kernel shape: [5, 5, 3]
        # Output shape: [batch_size, noise_w*2, noise_h*2, filters=256]
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=256, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob) #2
        # Second batch norm layer
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)

        # Second convolutional layer: applies 128 different kernels over each "image" from the previous layer, with stride of 2
        # Kernel shape: [5, 5, 256]
        # Output shape: [batch_size, noise_w*4=original_w, noise_h*4=original_h, filters=128]
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=128, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob) #3
        # Third batch norm layer
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)

        # Third convolutional layer: applies 64 different kernels over each "image" from the previous layer, with stride of 1
        # Kernel shape: [5, 5, 128]
        # Output shape: [batch_size, noise_w*4=original_w, noise_h*4=original_h, filters=64]
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob) #4
        # Fourth batch norm layer
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)

        # Fourth convolutional layer: applied 3 different kernels over each "image" from the previous layer, with stride of 1
        # Kernel shape: [5, 5, 64]
        # Output shape: [batch_size, noise_w*4=original_w, noise_h*4=original_h, filters=c]
        # The resulting output shape is now indentical to that of our original images [w,h,c] where c is 3 for RGB images
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=c, strides=1, padding='same', activation=tf.nn.sigmoid)

        # Returns
        return x

# initialize generator object
g = generator(noise, keep_prob, is_training)
# print(g)

# initialize 2 discrimininator obejcts, d_real will be fed real images and d_fake will be fed the generated images
d_real = discriminator(X_in)
d_fake = discriminator(g, reuse=True) # reuse=True ensures that the weights are the same for both discrimininator objects

# define 2 different variable scopes, one for the generator and one for the disrciminator
vars_g = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]

# initialize l2 regularization functions, one each for generator and discriminator
# will be passed to the optimizer along with the loss functions
d_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_d)
g_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_g)

# discriminator has two loss functions, one for each disrciminator object (d_real and d_fake)
# loss_d_real measures how often the discriminator correctly classifies real data as real
loss_d_real = binary_cross_entropy(tf.ones_like(d_real), d_real)
# loss_d_fake measures how often the discriminator correctly classifies fake data as fake
loss_d_fake = binary_cross_entropy(tf.zeros_like(d_fake), d_fake)
# loss_g measures how often the discriminator incorrectly classified fake data as real
loss_g = tf.reduce_mean(binary_cross_entropy(tf.ones_like(d_fake), d_fake))

# get average of loss_d_real and loss_d_fake for an overall loss function for discriminator
loss_d = tf.reduce_mean(0.5 * (loss_d_real + loss_d_fake))

# get update_ops and add as control dependency (required for moving average of means and std dev's in batch_norm)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # minimize the discriminator loss plus the l-2 regularization of discriminator's weights
    optimizer_d = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(loss_d + d_reg, var_list=vars_d)
    # minimize the generator loss plus the l-2 regularization of generator's weights
    optimizer_g = tf.train.RMSPropOptimizer(learning_rate=0.0002).minimize(loss_g + g_reg, var_list=vars_g)

# initialize tf sessions and load the global variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# training
print("training")
num_iterations = 5000
catch_up_iterations = 5

loss_log_filename = "loss_log_homer_5.csv"
loss_log = open(loss_log_filename, "w")
loss_log.write("d_loss,g_loss,train_d,train_g,catch_up\n")

for i in range(num_iterations):
    # set both discriminator and generator to be trained simultaneously
    train_d = True
    train_g = True
    catch_up = False
    keep_prob_train = 0.6 # 0.5

    # generate a batch of noise vectors, to be input into generator
    n = np.random.uniform(0.0, 1.0, [batch_size, n_noise]).astype(np.float32)

    # generate a random batch of real images, save as a list of numpy arrays
    batch = [b for b in next_batch(num=batch_size)]

    # run one pass through the networks, passing both real images and the noise vectors
    d_real_ls, d_fake_ls, g_ls, d_ls = sess.run([loss_d_real, loss_d_fake, loss_g, loss_d], feed_dict={X_in: batch, noise: n, keep_prob: keep_prob_train, is_training:True})

    # d_fake_ls_init = d_fake_ls #pretty sure this is unnecessary, is never referenced again
    #
    # d_real_ls = np.mean(d_real_ls)
    # d_fake_ls = np.mean(d_fake_ls)
    # g_ls = g_ls
    # d_ls = d_ls
    loss_log.write("{0},{1},{2},{3},{4}\n".format(str(d_ls),str(g_ls),str(train_d),str(train_g),str(catch_up)))

    # if loss of the discriminator is greater than  1.35 times the loss of the generator, stop training the generator for now
    if g_ls * 1.35 < d_ls:
        train_g = False
        pass
    # if loss of the generator is greater than 1.35 times the loss of the discriminator, stop training the discriminator for now
    if d_ls * 1.20 < g_ls:
        train_d = False
        pass

    # # run a second pass, allowing either the discriminator or generator to catch up
    # if train_d:
    #     sess.run(optimizer_d, feed_dict={noise: n, X_in: batch, keep_prob: keep_prob_train, is_training:True})
    #
    # if train_g:
    #     sess.run(optimizer_g, feed_dict={noise: n, keep_prob: keep_prob_train, is_training:True})

    if train_d and train_g:
        sess.run(optimizer_d, feed_dict={noise: n, X_in: batch, keep_prob: keep_prob_train, is_training:True})
        sess.run(optimizer_g, feed_dict={noise: n, keep_prob: keep_prob_train, is_training:True})

    if not train_d and train_g:
        catch_up = True
        for it in range(catch_up_iterations):
            d_ls,g_ls,opt = sess.run([loss_d,loss_g,optimizer_g], feed_dict={X_in: batch, noise: n, keep_prob: keep_prob_train, is_training:True})
            loss_log.write("{0},{1},{2},{3},{4}\n".format(str(d_ls),str(g_ls),str(train_d),str(train_g),str(catch_up)))

    if train_d and not train_g:
        sess.run([loss_d,loss_g,optimizer_d], feed_dict={noise: n, X_in: batch, keep_prob: keep_prob_train, is_training:True})
        loss_log.write("{0},{1},{2},{3},{4}\n".format(str(d_ls),str(g_ls),str(train_d),str(train_g),str(catch_up)))

    # print progress output
    if not i % 25:
        print('Iter: {}'.format(i))
        print('Discriminator loss: {:.4}'.format(d_ls))
        print('Generator loss: {:.4}'.format(g_ls))

        if not train_g:
            print("not training generator")
        if not train_d:
            print("not training discriminator")
        print()
        # get generator to see output images
        gen_imgs, discrim_scores = sess.run([g, d_fake], feed_dict = {noise: n, keep_prob: 1.0, is_training:False})
        imgs = best_gen_imgs(np.array(gen_imgs),np.array(discrim_scores),16)
        imgs = [img[:,:,:] for img in gen_imgs]
        # create montage of 16 of the generated images
        m = montage(imgs[0:16])
        plt.axis('off')
        plt.imshow(m, cmap='gray')
        plt.savefig('{0}/{1}.png'.format(out_dir, str(i).zfill(5)), bbox_inches='tight')

loss_log.close()
