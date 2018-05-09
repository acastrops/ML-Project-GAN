import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import csv
import argparse
import numpy as np 
import scipy.misc
import sys
import math
import matplotlib.colors as colors

# make sure Xming is running
# export DISPLAY=localhost:0.0

tf.set_random_seed(0)
mb_size = 128
w, h = 48, 48
l = w*h
L = 1500
M = 1200
N = 900
O = 600
Z_dim = 300


# this is used to randomly assign weights
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, l]) # shape=[None, img_size, img_size, num_channels]
Z = tf.placeholder(tf.float32, shape=[None, Z_dim]) # shape=[None, num_fakes]

# variable learning rate
lr = tf.placeholder(tf.float32)
# step for variable learning rate
step = tf.placeholder(tf.int32)
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)

# Discriminator Layers
D_W1 = tf.Variable(xavier_init([l, mb_size]))
D_b1 = tf.Variable(tf.ones(shape=[mb_size]))

D_W2 = tf.Variable(xavier_init([mb_size, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

# Generator Layers
# five layers and their number of neurons (tha last layer has 10 softmax neurons)

# Weights initialised with small random values between -0.2 and +0.2
# When using sigmoids, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
G_W1 = tf.Variable(tf.abs(tf.truncated_normal([Z_dim, O]))) # 784 = 28 * 28
G_B1 = tf.Variable(tf.zeros([O])/Z_dim)

G_W2 = tf.Variable(tf.abs(tf.truncated_normal([O, N])))
G_B2 = tf.Variable(tf.zeros([N])/Z_dim)

G_W3 = tf.Variable(tf.abs(tf.truncated_normal([N, M])))
G_B3 = tf.Variable(tf.zeros([M])/Z_dim)

G_W4 = tf.Variable(tf.abs(tf.truncated_normal([M, L])))
G_B4 = tf.Variable(tf.zeros([L])/Z_dim)

G_W5 = tf.Variable(tf.abs(tf.truncated_normal([L, l])))
G_B5 = tf.Variable(tf.zeros([l]))

theta_G = [G_W1, G_W2, G_W3, G_W4, G_W5, G_B1, G_B2, G_B3, G_B4, G_B5]

# use to create noise
def sample_Z(m, n):
    noise = np.random.normal(size=[m, n])
    return noise



def generator(z):
    # The model
    # we could try leaky relu
    Y1 = tf.nn.sigmoid(tf.matmul(z, G_W1) + G_B1)
    Y1d = tf.nn.dropout(Y1, pkeep)
    # should we batch norm the layers?
    
    Y2 = tf.nn.sigmoid(tf.matmul(Y1d, G_W2) + G_B2)
    Y2d = tf.nn.dropout(Y2, pkeep)
   
    Y3 = tf.nn.sigmoid(tf.matmul(Y2d, G_W3) + G_B3)
    Y3d = tf.nn.dropout(Y3, pkeep)
    
    Y4 = tf.nn.sigmoid(tf.matmul(Y3d, G_W4) + G_B4)
    Y4d = tf.nn.dropout(Y4, pkeep)
    
    G_log_prob = tf.matmul(Y4d, G_W5) + G_B5
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.sigmoid(tf.matmul(x, D_W1) + D_b1)
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
        plt.imshow(sample.reshape(w, h), cmap='gray')

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
D_loss = (D_loss_real + D_loss_fake)
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

# training step
# the learning rate is: # 0.0001 + 0.003 * (1/e)^(step/2000)), i.e. exponential decay from 0.003->0.0001
lr = 0.0001 +  tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(.00001).minimize(G_loss, var_list=theta_G)


image = np.zeros((l), dtype=np.uint8)
id = 0
list3d = [0]*8990
with open('../fer2013/fer2013.csv', 'rt') as csvfile:
    datareader = csv.reader(csvfile, delimiter =',')
    headers = next(datareader, None)
    for row in datareader:

        emotion = row[0]
        if emotion != '3':
            continue
        # if id == 1000:
        #     break
        pixels = row[1].split()
        usage = row[2]

        pixels_array = np.asarray(pixels).astype(np.float)
        pixels_array = np.divide(pixels_array, 255.)
        image = pixels_array
        
        list3d[id] = image
        id += 1 
        if id % 100 == 0:
            print('Processed {} images'.format(id))

list3d = list3d[0:id]
print("Finished processing {} images".format(id))
array3d = np.vstack(list3d)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim), pkeep: 1.0})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_mb = array3d[np.random.choice(array3d.shape[0], mb_size, replace=True), :]
    
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim), pkeep: 0.66})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim), step: i, pkeep: 0.66})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()