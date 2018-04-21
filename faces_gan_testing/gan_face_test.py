import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pandas as pd

def prep_input(pixel_str):
    pixel_list = pixel_str.split(' ')
    pixel_array = np.array(pixel_list, dtype = 'float64')
    return pixel_array

# read in csv of labels and pixels in to dataframe
df = pd.read_csv('fer2013.csv')
df.pixels = df.pixels.apply(prep_input)
happy_df = df[df.emotion == 3]
# print(happy_df.tail())

faces = np.array(list(happy_df.pixels))

# faces_tensor = tf.convert_to_tensor(faces)
# faces_dataset = tf.data.Dataset.from_tensor_slices(faces_tensor)
# faces2 = np.reshape(faces,(35887, 2034))
print(faces.shape)

def next_batch(num, data, total,labels = None):

    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.random.choice(total, num, replace = False)
    rand_batch = data[idx,:]
    # labels_shuffle = labels[idx]
    # labels_shuffle = np.asarray(labels_shuffle.values.reshape(len(labels_shuffle), 1))

    return rand_batch #, labels_shuffle


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

pixels_per_img = 2304
mb_size = 128
Z_dim = 100
num_imgs = 35887


X = tf.placeholder(tf.float32, shape=[None, pixels_per_img]) # shape=[None, img_size, img_size, num_channels]
Z = tf.placeholder(tf.float32, shape=[None, Z_dim]) # shape=[None, num_classes]? is this y - true?

# Discriminator Layers
D_W1 = tf.Variable(xavier_init([pixels_per_img, mb_size]))
D_b1 = tf.Variable(tf.zeros(shape=[mb_size]))

D_W2 = tf.Variable(xavier_init([mb_size, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

# Generator Layers
G_W1 = tf.Variable(xavier_init([Z_dim, mb_size]))
G_b1 = tf.Variable(tf.zeros(shape=[mb_size]))

G_W2 = tf.Variable(xavier_init([mb_size, pixels_per_img]))
G_b2 = tf.Variable(tf.zeros(shape=[pixels_per_img]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(48, 48), cmap='Greys_r')

    return fig


G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)




sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('faces_out/'):
    os.makedirs('faces_out/')

i = 0

for it in range(1000000):
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})

        fig = plot(samples)
        plt.savefig('faces_out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_mb = next_batch(mb_size, faces, num_imgs)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
