# -*- coding: utf-8 -*-
import tensorflow as tf
from libs import utils
import numpy as np
import os
from PIL import Image

def create_input_pipline(files,batch_size,shape,
                         n_threads=8,is_train=True):
    #create filename queue
    filename_queue=tf.train.string_input_producer(files)
    
    
    reader=tf.TFRecordReader()
    _,serialized_example=reader.read(filename_queue)
    
    
    #get feature from serialized example
    features=tf.parse_single_example(
        serialized_example,
        features={
            'img':tf.FixedLenFeature([],tf.string)
        }
    )
    
    img_raw=features['img']
    
    img=tf.decode_raw(img_raw,tf.uint8)  #从二进制解码
    img=tf.reshape(img,[64,64,3])        #指定图片的形状
    
    reshape_img=tf.cast(img,tf.float32)/255   #将数据缩小到0-1范围内。
    
    min_after_dequeue=20000
    capacity = min_after_dequeue + (n_threads + 1) * batch_size  #设置样本容量

    # Randomize the order and output batches of batch_size.
    batch_imgs= tf.train.shuffle_batch([reshape_img],
                                   enqueue_many=False,
                                   batch_size=batch_size,
                                   capacity=capacity,
                                   min_after_dequeue=min_after_dequeue,
                                   num_threads=n_threads)    
    
    return batch_imgs

def generator(z,phase_train=False,output_h=32,output_w=32,n_features=64,reuse=False):
    """
     output_h:Final generated height
    """
    with tf.variable_scope("generator",reuse=reuse) as scope:
      with tf.variable_scope("0",reuse=reuse):
        z_,W_=utils.linear(z,n_features*8*4*4,name='g_h0_lin')
        
        h0=tf.reshape(z_,[-1,4,4,n_features*8])
        norm0=utils.batch_norm()
        norm0_1=norm0(h0,phase_train)
        h0=tf.nn.relu(norm0_1)
        
      with tf.variable_scope("1",reuse=reuse):
        h1,W1=utils.deconv2d(h0,n_features//8,n_features//8,n_features*4,name='g_h1')
        norm1=utils.batch_norm()
        norm1_1=norm1(h1,phase_train)
        h1=tf.nn.relu(norm1_1)
        
      with tf.variable_scope("2",reuse=reuse):
        h2,W2=utils.deconv2d(h1,n_features//4,n_features//4,n_features*2,name='g_h2')
        norm2=utils.batch_norm()
        norm2_1=norm1(h2,phase_train)
        h2=tf.nn.relu(norm2_1)
        
      with tf.variable_scope("3",reuse=reuse):
        h3,W3=utils.deconv2d(h2,n_features//2,n_features//2,n_features*1,name='g_h3')
        norm3=utils.batch_norm()
        norm3_1=norm1(h3,phase_train)
        h3=tf.nn.relu(norm3_1)
        
      with tf.variable_scope("4",reuse=reuse):
        h4,W4=utils.deconv2d(h3,output_h,output_w,3,name='g_h4')

        
    return tf.nn.tanh(h4)
    
def discriminator(x,phase_train=False,n_features=64,reuse=False):
    """
    dimensions:
    """
    with tf.variable_scope("discriminator",reuse=reuse):
       with tf.variable_scope("0",reuse=reuse): 
        conv0,W0=utils.conv2d(x,n_features,name='d_h0_conv')
        h0=utils.lrelu(conv0)
       
       with tf.variable_scope("1",reuse=reuse): 
        conv1,W1=utils.conv2d(h0,n_features*2,name='d_h1_conv')
        norm1=utils.batch_norm()
        norm1_1=norm1(conv1,phase_train)
        h1=utils.lrelu(norm1_1)
      
       with tf.variable_scope("2",reuse=reuse):  
        conv2,W2=utils.conv2d(h1,n_features*4,name='d_h2_conv')
        norm2=utils.batch_norm()
        norm2_1=norm2(conv2,phase_train)
        h2=utils.lrelu(norm2_1)
       
       with tf.variable_scope("3",reuse=reuse):  
        conv3,W3=utils.conv2d(h2,n_features*8,name='d_h3_conv')
        norm3=utils.batch_norm()
        norm3_1=norm3(conv3,phase_train)
        h3=utils.lrelu(norm3_1)
        
       with tf.variable_scope("4",reuse=reuse):
        h4,W4=utils.linear(tf.reshape(h3,[-1,8192]),n_output=1,name='d_h3_lin')
        
    return tf.nn.sigmoid(h4),h4

    
def GAN(input_shape,n_latent,n_features,lam=0.1):
    #complete the image
    mask=tf.placeholder(tf.float32,input_shape,name='mask')
    
    #Real input samples
    #n_features is either the image dimension
    x=tf.placeholder(tf.float32,input_shape,'x')
    sum_x=tf.summary.image("x",x)
    phase_train=tf.placeholder(tf.bool,name='phase_train')
    
    #Discriminator for real input samples
    D_real,D_real_logits=discriminator(x,phase_train,n_features=n_features)
    sum_D_real=tf.summary.histogram("D_real",D_real)
    
    #Generator tries to recreate input samples using latent feature vector
    z=tf.placeholder(tf.float32,[None,n_latent],'z')
    sum_z=tf.summary.histogram("z",z)
    G=generator(z,phase_train,
                output_h=input_shape[1],output_w=input_shape[2],
                n_features=n_features)
    sum_G=tf.summary.image("G",G)
    
    #Discriminator for generated samples
    D_fake,D_fake_logits=discriminator(G,phase_train,
                                       n_features=n_features,reuse=True)
    sum_D_fake=tf.summary.histogram("D_fake",D_fake)
    
    with tf.variable_scope("loss"):
        #loss function
        loss_D_real=tf.nn.sigmoid_cross_entropy_with_logits(D_real_logits,
                                                            tf.ones_like(D_real),name='loss_D_real')
        loss_D_fake=tf.nn.sigmoid_cross_entropy_with_logits(D_fake_logits,
                                                            tf.zeros_like(D_fake),name='loss_D_fake')
        loss_D=tf.reduce_mean(loss_D_real+loss_D_fake)
        
        loss_G=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                             D_fake_logits,tf.ones_like(D_fake),name='loss_G'))
        
        #Summaries
        sum_loss_D_real = tf.summary.histogram("loss_D_real", loss_D_real)
        sum_loss_D_fake = tf.summary.histogram("loss_D_fake", loss_D_fake)
        sum_loss_D = tf.summary.scalar("loss_D", loss_D)
        sum_loss_G = tf.summary.scalar("loss_G", loss_G)
        sum_D_real = tf.summary.histogram("D_real", D_real)
        sum_D_fake = tf.summary.histogram("D_fake", D_fake)
        
    with tf.variable_scope("complete_loss"):
        contextual_loss=tf.reduce_sum(
                tf.contrib.layers.flatten(
                    tf.abs(tf.mul(mask,G)-tf.mul(mask,x))),1)
        perceptual_loss=loss_G
        
        complete_loss=contextual_loss+lam*perceptual_loss
        grad_complete_loss=tf.gradients(complete_loss,z)
    
    return {
            'loss_D':loss_D,
            'loss_G':loss_G,
            'x':x,
            'G':G,
            'z':z,
            'train':phase_train,
            'sums':{
                'G':sum_G,
                'D_real':sum_D_real,
                'D_fake': sum_D_fake,
                'loss_G': sum_loss_G,
                'loss_D': sum_loss_D,
                'loss_D_real': sum_loss_D_real,
                'loss_D_fake': sum_loss_D_fake,
                'z': sum_z,
                'x': sum_x   
            },
            'complete':{
                'mask':mask,
                'complete_loss':complete_loss,
                'grad_complete_loss':grad_complete_loss
            }
    }
    
def train_gan():
    batch_size=64
    sample_size=64
    n_latent=100
    lr_g=0.0002
    lr_d=0.0002
    data_dir='data/train.tfrecord'
    
    #sample
    filename=data_dir
    batch_img=create_input_pipline([filename],batch_size=batch_size,shape=[64,64,3])
    zs=np.random.uniform(-1,1,size=(batch_size,n_latent))
    
    #model
    gan=GAN(input_shape=[None,64,64,3],n_features=64,
            n_latent=100)
    
    #optimize
    vars_d=[v for v in tf.trainable_variables()
            if 'd_' in v.name]
    vars_g=[v for v in tf.trainable_variables()
            if 'g_' in v.name]
            
    opt_g=tf.train.AdamOptimizer(lr_g,name='Adam_g').minimize(
        gan['loss_G'],var_list=vars_g)
    opt_d=tf.train.AdamOptimizer(lr_d,name='Adam_d').minimize(
        gan['loss_D'],var_list=vars_d)
    
    #create session to use the graph
    sess=tf.Session()
    init_op=tf.global_variables_initializer()
    
    saver=tf.train.Saver()
   
    sums=gan['sums']
    G_sum_op=tf.summary.merge([
                sums['G'],sums['loss_G'],sums['z'],
                sums['loss_D_fake'],sums['D_fake']])
    D_sum_op = tf.summary.merge([
        sums['loss_D'], sums['loss_D_real'], sums['loss_D_fake'],
        sums['z'], sums['x'], sums['D_real'], sums['D_fake']])
    
    writer = tf.summary.FileWriter("./train_dir/logs", sess.graph)
    
    sess.run(init_op)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    # g = tf.get_default_graph()
    # [print(op.name) for op in g.get_operations()]

    ckpt = tf.train.get_checkpoint_state('train_dir/checkpoint')
    if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("GAN model restored")
    
    print('start train')
    for step_i in range(500001):
         batch_images=sess.run(batch_img)
         batch_zs=np.random.uniform(-1,1,size=(batch_size,n_latent))
         
#         print('start update D network')
         #update D network
         loss_d,_,summary_str=sess.run([gan['loss_D'],opt_d,D_sum_op],
                                       feed_dict={gan['x']:batch_images,gan['z']:batch_zs,
                                                  gan['train']:True})
         writer.add_summary(summary_str,step_i)
         
#         print('start update G network')
         #update G network
         loss_g,_,summary_str=sess.run([gan['loss_G'],opt_g,G_sum_op],
                                       feed_dict={gan['z']:batch_zs,gan['train']:True})
         writer.add_summary(summary_str,step_i)
         #ran g_optim twice
         loss_g,_,summary_str=sess.run([gan['loss_G'],opt_g,G_sum_op],
                                       feed_dict={gan['z']:batch_zs,gan['train']:True})
         writer.add_summary(summary_str,step_i)
         
         if step_i % 100==0:
             print(step_i,'loss:',loss_g+loss_d)
             samples = sess.run(gan['G'], feed_dict={
                    gan['z']: zs,
                    gan['train']: False})
             utils.montage(np.clip((samples + 1) * 127.5, 0, 255).astype(np.uint8),
                        'train_dir/imgs/gan_%08d.png' % step_i)     
        
         if step_i % 1000==0:
            save_path = saver.save(sess, "./train_dir/checkpoint/gan.ckpt",
                                       global_step=step_i,
                                       write_meta_graph=False)
            print("Model saved in file: %s" % save_path)
         
            
    coord.request_stop()
    coord.join(threads) 
     
    sess.close()

def completion():
    batch_size=64
    n_latent=100
    learning_rate=0.01
    beta1=0.9
    beta2=0.999
    eps=1e-8
    maskType='center'
    input_shape=[64,64,3]
    
    #sample
    zs=np.random.uniform(-1,1,size=(batch_size,n_latent))
    
    #model
    gan=GAN(input_shape=[None,64,64,3],n_features=64,
            n_latent=100)


    if maskType == 'center':
            scale = 0.25
            assert(scale <= 0.5)
            mask = np.ones(input_shape)
            l = int(input_shape[0]*scale)
            u = int(input_shape[0]*(1.0-scale))
            mask[l:u, l:u, :] = 0.0
    
    data_dir='data/crop_images_DB/'
    train_imgpaths=[os.path.join(data_dir,img_i) 
                for img_i in os.listdir(data_dir)]
    imgpaths=train_imgpaths[0:batch_size]
    imgs=[]
    for img_path in imgpaths:
        with Image.open(img_path) as img:
            arr_img = np.asarray(img, dtype='uint8')/255
            imgs.append(arr_img)
    
    batch_images=np.asarray(imgs).astype(np.float32)
    batch_mask=np.resize(mask,[batch_size]+input_shape)
    zhats = np.random.uniform(-1, 1, size=(batch_size,n_latent))
    m = 0
    v = 0    
    
    utils.montage(np.clip((batch_images + 1) * 127.5, 0, 255).astype(np.uint8),
                        'complete_dir/imgs/before.png' )
    masked_images=np.multiply(batch_images,batch_mask)
    utils.montage(np.clip((masked_images+1)*127.5, 0, 255).astype(np.uint8),
                        'complete_dir/imgs/masked.png' )
    
    
    sess=tf.Session()
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    
    saver=tf.train.Saver()
    
    # g = tf.get_default_graph()
    # [print(op.name) for op in g.get_operations()]

    ckpt = tf.train.get_checkpoint_state('train_dir/checkpoint')
    if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("GAN model restored")
    
    print('start complete')
    complete=gan['complete']
    

    for i in range(1001):
        loss,g,G_imgs=sess.run([complete['complete_loss'],complete['grad_complete_loss'],gan['G']],
                               feed_dict={gan['z']:zhats,complete['mask']:batch_mask,gan['x']:batch_images,gan['train']:False})
        
        
#        v_prev = np.copy(v)
#        v = m*v - learning_rate
#        zhats += -m * v_prev + (1+m)*v
#        zhats = np.clip(zhats, -1, 1)

        m_prev = np.copy(m)
        v_prev = np.copy(v)
        m = beta1 * m_prev + (1 - beta1) * g[0]
        v = beta2 * v_prev + (1 - beta2) * np.multiply(g[0], g[0])
        m_hat = m / (1 - beta1 ** (i + 1))
        v_hat = v / (1 - beta2 ** (i + 1))
        zhats +=  -np.true_divide(learning_rate * m_hat, (np.sqrt(v_hat) + eps))
        zhats = np.clip(zhats, -1, 1)
        
        if i % 100 == 0:
            print(np.mean(loss))
            utils.montage(np.clip((G_imgs + 1) * 127.5, 0, 255).astype(np.uint8),
                        'complete_dir/imgs/hats_imgs/{:04d}.png'.format(i) )

            inv_masked_hat_images = np.multiply(G_imgs, 1.0-batch_mask)
            completeed = (masked_images + inv_masked_hat_images)/2
            utils.montage(np.clip((completeed + 1) * 127.5, 0, 255).astype(np.uint8),
                        'complete_dir/imgs/completed/{:04d}.png'.format(i) )
    sess.close()
    
    return G_imgs,completeed

    
if __name__ == '__main__':
   #train_gan()
    a,b=completion()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    