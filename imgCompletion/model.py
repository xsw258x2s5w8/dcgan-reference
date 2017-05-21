# -*- coding: utf-8 -*-
import tensorflow as tf
from libs.utils import *
from six.moves import xrange
from vec import vectorize_imgs
import numpy as np
import os

class DCGAN(object):
    def __init__(self,sess,image_size=64,sample_size=64,
                 z_dim=100,df_dim=64,gf_dim=64,batch_size=64,
                 data_dir='data/crop_images_DB/',checkpoint_dir=None):
        """
        Args:
            sess:Tensorflow session
            z_dim:(optional)Dimension of dim for z.[100]
            df_dim:(optional )Dimension of discrim filters in first con layer
            gf_dim:(optional )Dimension of gen filters in first con layer
        """
        self.sess=sess
        self.image_size=image_size
        self.sample_size=sample_size
        self.image_shape=[image_size,image_size,3]
        self.batch_size=batch_size
        
        self.data_dir=data_dir
        self.checkpoint_dir=checkpoint_dir
        
        self.z_dim=z_dim
        
        self.df_dim=df_dim
        self.gf_dim=gf_dim
        
        #batch normalization
        self.d_bn1=batch_norm(name='d_bn1')
        self.d_bn2=batch_norm(name='d_bn2')
        self.d_bn3=batch_norm(name='d_bn3')
        
        self.g_bn0=batch_norm(name='g_bn0')
        self.g_bn1=batch_norm(name='g_bn1')
        self.g_bn2=batch_norm(name='g_bn2')
        self.g_bn3=batch_norm(name='g_bn3')
        
        
    def build_model(self):
        self.is_training=tf.placeholder(tf.bool,name='is_training')
        self.images=tf.placeholder(tf.float32,[None]+self.image_shape,name='real_images')
        self.sample_images=tf.placeholder(tf.float32,[None]+self.image_shape,name='sample_images')
        self.z=tf.placeholder(tf.float32,[None,self.z_dim],name='z')
        self.z_sum=tf.summary.histogram("z",self.z)
        
        self.G=self.generator(self.z)
        
        self.D,self.D_logits=self.discriminator(self.images)
        self.D_,self.D_logits_=self.discriminator(self.G,reuse=True)
        
        self.d_sum=tf.summary.histogram("d",self.D)
        self.d__sum=tf.summary.histogram("d_",self.D_)
        self.G_sum=tf.summary.image("G",self.G)
        
        #loss
        self.d_loss_real=tf.reduce_mean(
             tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                                     targets=tf.ones_like(self.D)))
        
        self.d_loss_fake=tf.reduce_mean(
             tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                     targets=tf.zeros_like(self.D_)))
        
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        
        self.d_loss=self.d_loss_real+self.d_loss_fake
        
        self.g_loss=tf.reduce_mean(
             tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                     targets=tf.ones_like(self.D_)))
        
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        
        
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

    def train(self,config):
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)  
    
        tf.global_variables_initializer().run()
        
        self.g_sum = tf.summary.merge(
            [self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        
        counter=1
        
        if self.load(self.checkpoint_dir):
            print('Load checkpoint finish')
        else:
            print('Initializing a new one')
        
        for epoch_i in xrange(20):
            dataList=[os.path.join(self.data_dir,img_i) 
                for img_i in os.listdir(self.data_dir)]
            batch_idxs=len(dataList)//self.batch_size
            
            for idx in xrange(0,batch_idxs):
                batch_files=dataList[idx*self.batch_size:(idx+1)*self.batch_size]
                batch=[vectorize_imgs(img_i) 
                        for img_i in batch_files]
                batch_imgs=np.array(batch).astype(np.float32)
                
                batch_z=np.random.uniform(-1,1,[self.batch_size,self.z_dim]).astype(np.float32)
                
                #update D network
                _,summary_str=self.sess.run([d_optim,self.d_sum],
                    feed_dict={self.images:batch_imgs,self.z:batch_z,self.is_training:True})
                self.writer.add_summary(summary_str,counter)
                
                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                    feed_dict={ self.z: batch_z, self.is_training: True })
                self.writer.add_summary(summary_str, counter)
                
                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                    feed_dict={ self.z: batch_z, self.is_training: True })
                self.writer.add_summary(summary_str, counter)
                
                errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.is_training: False})
                errD_real = self.d_loss_real.eval({self.images: batch_imgs, self.is_training: False})
                errG = self.g_loss.eval({self.z: batch_z, self.is_training: False})
                
                counter+=1
                print(errD_fake,errD_real,errG)
                
                if counter%1000==0:
                    self.save(self.checkpoint_dir,counter)
        
    def discriminator(self,image,reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
                
            conv0,W0=conv2d(image, self.df_dim, name='d_h0_conv0')
            h0=lrelu(conv0)
            
            conv1,W1=conv2d(h0,self.df_dim*2,name='d_h1_conv1')
            d_bn1=self.d_bn1(conv1,self.is_training)
            h1=lrelu(conv1)
            
            conv2,W2=conv2d(h1,self.df_dim*2,name='d_h2_conv2')
            d_bn2=self.d_bn2(conv2,self.is_training)
            h2=lrelu(conv2)
            
            conv3,W3=conv2d(h2,self.df_dim*2,name='d_h1_conv3')
            d_bn3=self.d_bn3(conv3,self.is_training)
            h3=lrelu(conv3)
            
            h4,W4=linear(tf.reshape(h3,[-1,8192]),1,'d_h3_fc')
            
            return tf.nn.sigmoid(h4),h4
            
    def generator(self,z):
        with tf.variable_scope("generator") as scope:
            self.z_,self.h0_2,self.h0_b=linear(z,self.gf_dim*8*4*4,name='g_h0_fc',with_w=True)
            
            self.h0=tf.reshape(self.z_,[-1,4,4,self.gf_dim*8])
            g_bn0=self.g_bn0(self.h0,self.is_training)
            h0=tf.nn.relu(g_bn0)
            
            self.h1,self.h1_w,self.h1_b=deconv2d(h0,8,8,self.gf_dim*4,name='g_h1')
            g_bn1=self.g_bn1(self.h1,self.is_training)
            h1=tf.nn.relu(g_bn1)
            
            h2,self.h2_w,self.h2_b=deconv2d(h1,16,16,self.gf_dim*2,name='g_h2')
            g_bn2=self.g_bn2(h2,self.is_training)
            h2=tf.nn.relu(g_bn2)
            
            h3,self.h3_w,self.h3_b=deconv2d(h2,32,32,self.gf_dim*1,name='g_h3')
            g_bn3=self.g_bn3(h3,self.is_training)
            h1=tf.nn.relu(g_bn3)
            
            h4,self.h4_w,self.h4_b=deconv2d(h0,64,64,3,name='g_h4')

            return tf.nn.tanh(h4)
            
    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            