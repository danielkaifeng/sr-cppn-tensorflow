import os
import time
import numpy as np
import tensorflow as tf
from ops import *
from math import atan2
from vggnet import osvos, osvos_arg_scope, interp_surgery
slim = tf.contrib.slim
#from utils import get_assignment_map_from_checkpoint 

'''
https://github.com/carpedm20/DCGAN-tensorflow
https://jmetzen.github.io/2015-11-27/vae.html
https://en.wikipedia.org/wiki/Compositional_pattern-producing_network
'''

coord_dim = 6

class CPPNVAE():
  def __init__(self, batch_size=1, z_dim=32,
                x_dim = 26, y_dim = 26, c_dim = 1, scale = 8.0,
                learning_rate= 0.01, learning_rate_d= 0.001, learning_rate_vae = 0.0001, beta1 = 0.9, net_size_g = 128, net_depth_g = 6,
                net_size_q = 512, keep_prob = 1.0, df_dim = 24, model_name = "cppnvae"):
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.learning_rate_d = learning_rate_d
    self.learning_rate_vae = learning_rate_vae
    self.beta1 = beta1
    self.net_size_g = net_size_g
    self.net_size_q = net_size_q
    self.x_dim = x_dim
    self.y_dim = y_dim
    self.scale = scale
    self.c_dim = c_dim
    self.z_dim = z_dim
    self.net_depth_g = net_depth_g
    self.model_name = model_name
    self.keep_prob = keep_prob
    self.df_dim = df_dim

    graph = tf.get_default_graph()
    with graph.as_default():
    #with tf.Graph().as_default():
        self.input_image = tf.placeholder(tf.float32, [1, None, None, 3])
        with slim.arg_scope(osvos_arg_scope()):
            net, end_points = osvos(self.input_image)
        self.prob = tf.nn.sigmoid(net)


    # tf Graph batch of image (batch_size, height, width, depth)
    self.batch = tf.placeholder(tf.float32, [batch_size, x_dim, y_dim, c_dim], name='img')
    self.batch_flatten = tf.reshape(self.batch, [batch_size, -1])

    n_points = x_dim * y_dim
    self.n_points = n_points


    # latent vector
    # self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])
    # inputs to cppn, like coordinates and radius from centre
    self.x = tf.placeholder(tf.float32, [self.batch_size, None, 1], name='x')
    self.y = tf.placeholder(tf.float32, [self.batch_size, None, 1], name='y')
    self.r = tf.placeholder(tf.float32, [self.batch_size, None, 1], name='r')
    self.coord = tf.placeholder(tf.float32, [None, coord_dim], name='coord')

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(batch_size, name=self.model_name+'_d_bn1')
    self.d_bn2 = batch_norm(batch_size, name=self.model_name+'_d_bn2')

    self.G = self.generator(x_dim, y_dim)
    self.G = tf.reshape(self.G, [self.batch_size, y_dim, x_dim, 3])
    self.batch_reconstruct_flatten = tf.reshape(self.G, [batch_size, -1])

    self.D_right = self.discriminator(self.batch) # discriminiator on correct examples
    self.D_wrong = self.discriminator(self.G, reuse=True) # feed generated images into D

    self.define_opt()

    # Create a saver to load the network
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    # Launch the session
    #self.sess = tf.InteractiveSession()
    self.sess = tf.Session(config=config)
    checkpoint_path = "models/vgg/OSVOS_parent.ckpt-50000" 
    #var_list = [v for v in tf.trainable_variables() if 'osvos' in v.name]
    #init = tf.variables_initializer(var_list, name='init')

    init = tf.global_variables_initializer()
    self.sess.run(init)
    self.sess.run(interp_surgery(tf.global_variables()))
    
    self.saver = tf.train.Saver([v for v in tf.global_variables() if '-up' not in v.name and '-cr' not in v.name and 'osvos' in v.name])
    self.saver.restore(self.sess, checkpoint_path)

    #assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(var_list, checkpoint_path)
    #tf.train.init_from_checkpoint(checkpoint_path, assignment_map)
    print('partial init vgg')
    #self.saver = tf.train.Saver(tf.all_variables())

  def define_opt(self):
        self.loss_w = tf.placeholder(tf.float32, [self.batch_size, self.x_dim, self.y_dim], name='vgg_w')
        reconstr_loss = tf.squared_difference(self.G, self.batch)
        #reconstr_loss = tf.reduce_mean(tf.squared_difference(self.G, self.batch), axis=-1) * self.loss_w 
        #reconstr_loss = tf.squared_difference(self.G, self.batch) * tf.expand_dims(self.loss_w, 3)
        #self.z_mean, self.z_log_sigma_sq = self.encoder()
        # Draw one sample z from Gaussian distribution
        #eps = tf.random_normal((self.batch_size, self.z_dim), 0, 1, dtype=tf.float32)
        # z = mu + sigma*epsilon
        #self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        self.vae_loss = tf.reduce_mean(reconstr_loss) 
        #self.create_vae_loss_terms()
        self.create_gan_loss_terms()

        self.balanced_loss = 1.0 * self.g_loss + 1.0 * self.vae_loss # can try to weight these.

        self.t_vars = tf.trainable_variables()

        self.q_vars = [var for var in self.t_vars if (self.model_name+'_q_') in var.name]
        self.g_vars = [var for var in self.t_vars if (self.model_name+'_g_') in var.name]
        self.d_vars = [var for var in self.t_vars if (self.model_name+'_d_') in var.name]
        g_list = [var for var in self.t_vars if 'generator' in var.name]
        #self.vae_vars = self.q_vars+self.g_vars
        
        # Use ADAM optimizer
        #with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
            #self.d_opt = tf.train.AdamOptimizer(self.learning_rate_d, beta1=self.beta1) \
            #              .minimize(self.d_loss, var_list=self.d_vars)
            #self.g_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
            #              .minimize(self.balanced_loss, var_list=self.vae_vars)
            self.vae_opt = tf.train.AdamOptimizer(self.learning_rate_vae, beta1=self.beta1) \
                          .minimize(self.vae_loss, var_list = g_list)
            self.d_opt = self.g_opt = self.vae_opt


  def create_vae_loss_terms(self):
    # The loss is composed of two terms:
    # 1.) The reconstruction loss (the negative log probability
    #     of the input under the reconstructed Bernoulli distribution
    #     induced by the decoder in the data space).
    #     This can be interpreted as the number of "nats" required
    #     for reconstructing the input when the activation in latent
    #     is given.
    # Adding 1e-10 to avoid evaluatio of log(0.0)
    #reconstr_loss = -tf.reduce_sum(self.batch_flatten * tf.log(1e-10 + self.batch_reconstruct_flatten)
    #                   + (1-self.batch_flatten) * tf.log(1e-10 + 1 - self.batch_reconstruct_flatten), 1)
    #use mse
    #reconstr_loss = tf.reduce_sum(tf.square(self.batch_flatten - self.batch_reconstruct_flatten))
    reconstr_loss = tf.squared_difference(self.G, self.batch)
    self.vae_loss = tf.reduce_mean(reconstr_loss) 

    # 2.) The latent loss, which is defined as the Kullback Leibler divergence
    ##    between the distribution in latent space induced by the encoder on
    #     the data and some prior. This acts as a kind of regularizer.
    #     This can be interpreted as the number of "nats" required
    #     for transmitting the the latent space distribution given
    #     the prior.
    latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                       - tf.square(self.z_mean)
                                       - tf.exp(self.z_log_sigma_sq), 1)
    #self.vae_loss = tf.reduce_mean(reconstr_loss + latent_loss) / self.n_points # average over batch and pixel
    self.vae_loss = tf.reduce_mean(reconstr_loss) 

  def create_gan_loss_terms(self):
    # Define loss function and optimiser
    self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D_right), self.D_right)
    self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_wrong), self.D_wrong)
    self.d_loss = 1.0*(self.d_loss_real + self.d_loss_fake)/ 2.0
    self.g_loss = 1.0*binary_cross_entropy_with_logits(tf.ones_like(self.D_wrong), self.D_wrong)

  def coordinates2(self, x_dim = 32, y_dim = 32, scale = 1.0):
    n_pixel = x_dim * y_dim
    x_range = scale*(np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5
    y_range = scale*(np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5
    x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
    y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
    r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
    x_mat = np.tile(x_mat.flatten(), self.batch_size).reshape(self.batch_size, n_pixel, 1)
    y_mat = np.tile(y_mat.flatten(), self.batch_size).reshape(self.batch_size, n_pixel, 1)
    r_mat = np.tile(r_mat.flatten(), self.batch_size).reshape(self.batch_size, n_pixel, 1)
    return x_mat, y_mat, r_mat

  def coordinates(self, x_dim, y_dim, scale = 1.0, vgg_w=None):
        X = []
        n_pixel = x_dim * y_dim
        for x_it in range(0, int(x_dim * scale)):
            for y_it in range(0, int(y_dim * scale)):
                xi = int(x_it/scale)
                yi = int(y_it/scale)
                w = vgg_w[xi, yi]
                #if vgg_w[x_it, y_it] == 0:
                #    x0=y0=x=y=0
                #    print('--------->', x_it, y_it)
                if 1:
                    x0 = x_it / scale + 0.5
                    y0 = y_it / scale + 0.5
                    x = (x0 - x_dim / 2)
                    y = (y0 - y_dim / 2)

                #X.append((w, x0, y0, x_dim - x0, y_dim - y0, (x**2+y**2)**(1/2), atan2(y0, x0)))
                X.append((x0, y0, x_dim - x0, y_dim - y0, (x**2+y**2)**(1/2), atan2(y0, x0)))
                #Y.append(np.multiply(1/255, img[x_iterator][y_iterator]))
        X = np.asarray(X)
        return X


  def encoder(self):
    # Generate probabilistic encoder (recognition network), which
    # maps inputs onto a normal distribution in latent space.
    # The transformation is parametrized and can be learned.
    H1 = tf.nn.dropout(tf.nn.softplus(linear(self.batch_flatten, self.net_size_q, self.model_name+'_q_lin1')), self.keep_prob)
    H2 = tf.nn.dropout(tf.nn.softplus(linear(H1, self.net_size_q, self.model_name+'_q_lin2')), self.keep_prob)
    z_mean = linear(H2, self.z_dim, self.model_name+'_q_lin3_mean')
    z_log_sigma_sq = linear(H2, self.z_dim, self.model_name+'_q_lin3_log_sigma_sq') #/10.
    return (z_mean, z_log_sigma_sq)

  def discriminator(self, image, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    h0 = lrelu(conv2d(image, self.df_dim, name=self.model_name+'_d_h0_conv'))
    h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name=self.model_name+'_d_h1_conv')))
    h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name=self.model_name+'_d_h2_conv')))
    h3 = linear(tf.reshape(h2, [self.batch_size, -1]), 1, self.model_name+'_d_h2_lin')

    return tf.nn.sigmoid(h3)

  def generator(self, gen_x_dim, gen_y_dim, reuse = False):
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()

        #x = tf.concat([self.x, self.y, self.r], axis=2)
        x = self.coord
        x = tf.layers.dense(x, 128)
        #x = tf.nn.softplus(x)
        x = tf.layers.dense(x, 128)
        #x = tf.nn.softplus(x)

        for i in range(1, self.net_depth_g):
          x = tf.nn.relu(tf.layers.dense(x, 128))

        #output = tf.sigmoid(fully_connected(H, self.c_dim, self.model_name+'_g_'+str(self.net_depth_g)))
        output = tf.sigmoid(tf.layers.dense(x, 3))
        #result = tf.reshape(output, [self.batch_size, gen_y_dim, gen_x_dim, 3])

    return output


  def partial_train(self, batch, co_vec, vgg_weight):
    counter = 0

    for i in range(1):
      counter += 1
      _, vae_loss = self.sess.run((self.vae_opt, self.vae_loss),
                              #feed_dict={self.batch: batch, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec})
                              feed_dict={self.batch: batch, self.coord: co_vec, self.loss_w: vgg_weight})


    for i in range(0):
      counter += 1
      _, g_loss = self.sess.run((self.g_opt, self.g_loss),
                              feed_dict={self.batch: batch, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec})
      if g_loss < 0.6:
        break

    #d_loss = self.sess.run(self.d_loss,
    #                          feed_dict={self.batch: batch, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec})

    #if d_loss > 0.6 and g_loss < 0.75:
    #if d_loss > 0.1 or g_loss < 10:
    if 0:
      for i in range(1):
        counter += 1
        _, d_loss = self.sess.run((self.d_opt, self.d_loss),
                                feed_dict={self.batch: batch, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec})
        if d_loss < 0.6:
          break

    g_loss = d_loss = 0 # i just don't run gan opt
    return d_loss, g_loss, vae_loss, counter

  def encode(self, X):
      """Transform data by mapping it into the latent space."""
      # Note: This maps to mean of distribution, we could alternatively
      # sample from Gaussian distribution
      return self.sess.run(self.z_mean, feed_dict={self.batch: X})

  def generate(self, x_dim, y_dim, scale, coord):
    #z = np.random.normal(size=self.z_dim).astype(np.float32)
    #z = np.reshape(z, (self.batch_size, self.z_dim))

    #gen_x_vec, gen_y_vec, gen_r_vec = self.coordinates(x_dim, y_dim, scale = scale)
    #coord = self.coordinates(x_dim, y_dim, scale = scale)

    G = self.generator(gen_x_dim = x_dim, gen_y_dim = y_dim, reuse = True)
    G = tf.reshape(G, [self.batch_size, int(y_dim*scale), int(x_dim*scale), 3])
    
    #image = self.sess.run(G, feed_dict={self.z: z, self.x: gen_x_vec, self.y: gen_y_vec, self.r: gen_r_vec})
    image = self.sess.run(G, feed_dict={self.coord: coord})
    return image

  def save_model(self, checkpoint_path, epoch):
    """ saves the model to a file """
    self.saver.save(self.sess, checkpoint_path, global_step = epoch)

  def load_model(self, checkpoint_path):
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    print "loading model: ",ckpt.model_checkpoint_path

    #self.saver.restore(self.sess, checkpoint_path+'/'+ckpt.model_checkpoint_path)
    # use the below line for tensorflow 0.7
    self.saver.restore(self.sess, ckpt.model_checkpoint_path)

  def close(self):
    self.sess.close()


