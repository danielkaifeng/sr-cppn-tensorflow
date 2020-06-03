import numpy as np
import tensorflow as tf
from PIL import Image

import argparse
import time
import os
import cPickle

from model import CPPNVAE
from tfrc import get_input_data
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
from vggnet import test_init, preprocess_img

'''
compositional pattern-producing generative adversarial network

LOADS of help was taken from:
https://jmetzen.github.io/2015-11-27/vae.html
'''

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_epochs', type=int, default=100,
                                         help='training epochs')
    parser.add_argument('--display_step', type=int, default=20,
                                         help='display step')
    parser.add_argument('--checkpoint_step', type=int, default=1,
                                         help='checkpoint step')
    parser.add_argument('--batch_size', type=int, default=4,
                                         help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                                         help='learning rate for G and VAE')
    parser.add_argument('--learning_rate_vae', type=float, default=0.001,
                                         help='learning rate for VAE')
    parser.add_argument('--learning_rate_d', type=float, default=0.001,
                                         help='learning rate for D')
    parser.add_argument('--keep_prob', type=float, default=1.00,
                                         help='dropout keep probability')
    parser.add_argument('--beta1', type=float, default=0.65,
                                         help='adam momentum param for descriminator')
    parser.add_argument('--use_ckpt', action='store_true', default=False,
                                         help='whether to use the checkpoint')
    args = parser.parse_args()
    return args


s = 128
seq_len = 256 * 256 * 3
args = get_args()

#tf_filepath = "/data/ai-datasets/201-FFHQ/tfrecords/face1024.tfrecords"



def main():
    return train(args)

def train(args):
    tf_filepath = "/data/ai-datasets/201-FFHQ/tfrecords/face256.tfrecords"
    img = get_input_data(tf_filepath, seq_len, args.batch_size, 10)

    learning_rate = args.learning_rate
    learning_rate_d = args.learning_rate_d
    learning_rate_vae = args.learning_rate_vae
    batch_size = args.batch_size
    training_epochs = args.training_epochs
    display_step = args.display_step
    checkpoint_step = args.checkpoint_step # save training results every check point step
    beta1 = args.beta1
    keep_prob = args.keep_prob
    dirname = 'save'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(os.path.join(dirname, 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)

    n_samples = 10000

    x_dim = s
    y_dim = s
    z_dim = 32
    scale = 1
    cppn = CPPNVAE(batch_size=batch_size, 
            learning_rate = learning_rate, 
            learning_rate_d = learning_rate_d, 
            learning_rate_vae = learning_rate_vae, 
            beta1 = beta1, 
            keep_prob = keep_prob,
            x_dim = x_dim,
            y_dim = y_dim,
            c_dim = 3, 
            net_depth_g = 3,
            scale=scale
            )

    #cppn.define_opt()

    #checkpoint_path = "models/vgg/OSVOS_parent.ckpt-50000" 
    #with tf.Graph().as_default():
    #    with tf.device('/gpu:0'):
    #        sess, prob, input_image = test_init(checkpoint_path, cppn.sess)

    batch_images = cppn.sess.run(img)/255.
    batch_images = np.reshape(batch_images, [batch_size, 256, 256, 3])
    batch_images = np.array([cv2.resize(img, (s, s)) for img in batch_images])

    im = Image.fromarray(np.uint8(batch_images[0] * 255))
    outpath = "graph/0000.png" 
    im.save(outpath)

    #img = np.array(Image.open("stone.png"))
    image = preprocess_img(batch_images[0]*255)
    vgg_w = res = cppn.sess.run(cppn.prob, feed_dict={cppn.input_image: image})[:,:,:,0]
    #np.savetxt('feature2.txt', res, fmt='%.2f', delimiter=',')

    im = Image.fromarray(np.uint8(res[0] * 255))
    #outpath = "graph/0000_feature.png" 
    #im.save(outpath)
    
    plt.imshow(res[0])
    plt.axis('off')
    plt.show()
    exit(0)
    
    coord = cppn.coordinates(x_dim, y_dim, scale, vgg_w[0])
    #coord_reshape = np.reshape(coord, (1, x_dim * scale, y_dim * scale, 6))

    coordx8 = cppn.coordinates(x_dim, y_dim, 20, vgg_w[0])
   

    # load previously trained model if appilcable
    ckpt = tf.train.get_checkpoint_state(dirname)
    if args.use_ckpt:
        cppn.load_model(dirname)

    counter = 0
    for epoch in range(training_epochs):
        avg_d_loss = 0.
        avg_q_loss = 0.
        avg_vae_loss = 0.
        total_batch = int(n_samples / batch_size)

        for i in range(total_batch):
            d_loss, g_loss, vae_loss, n_operations = cppn.partial_train(batch_images, coord, vgg_w)

            if i % 100 == 0:
                gen = cppn.generate(x_dim=x_dim, y_dim=y_dim, scale=20, coord=coordx8)
                im = Image.fromarray(np.uint8(gen[0] * 255))
                outpath = "graph/%02d.png" % i
                im.save(outpath)

            # Display logs per epoch step
            if (counter+1) % display_step == 0:
                print "Sample:", '%d' % ((i+1)*batch_size), " Epoch:", '%d' % (epoch), \
                            "d_loss=", "{:.4f}".format(d_loss), \
                            "g_loss=", "{:.4f}".format(g_loss), \
                            "vae_loss=", "{:.4f}".format(vae_loss), \
                            "n_op=", '%d' % (n_operations)

            #z = np.random.normal(size=z_dim).astype(np.float32)
            #z = np.reshape(z, (batch_size, z_dim))
            #gen_img = cppn.sess.run(cppn.batch_reconstruct_flatten, feed_dict={cppn.z: z, cppn.x: x, cppn.y: y, cppn.r: r, cppn.batch: batch_images})
            #print(gen_img)


            counter += 1
            # Compute average loss
            avg_d_loss += d_loss / n_samples * batch_size
            avg_q_loss += g_loss / n_samples * batch_size
            avg_vae_loss += vae_loss / n_samples * batch_size

        # Display logs per epoch step
        if epoch >= 0:
            print "Epoch:", '%04d' % (epoch), \
                        "avg_d_loss=", "{:.6f}".format(avg_d_loss), \
                        "avg_q_loss=", "{:.6f}".format(avg_q_loss), \
                        "avg_vae_loss=", "{:.6f}".format(avg_vae_loss)

        # save model
        if epoch >= 0 and epoch % checkpoint_step == 0:
            checkpoint_path = os.path.join('save', 'model.ckpt')
            cppn.save_model(checkpoint_path, epoch)
            print "model saved to {}".format(checkpoint_path)


    # save model one last time, under zero label to denote finish.
    #cppn.save_model(checkpoint_path, 0)

if __name__ == '__main__':
    main()
