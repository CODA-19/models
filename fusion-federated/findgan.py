# Core libraries
import tensorflow as tf
import numpy as np

# Visualization
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.style.use('seaborn-whitegrid')

# GAN
from gan_utils import uniform_sampler, binary_sampler, sample_batch_index
from topology import MaskGenerator, MaskDiscriminator, DataGenerator
from topology import DataDiscriminator, Encoder, Decoder

# Embedding / clustering
#from hdbscan import HDBSCAN
#from umap import UMAP

class FindGAN:

  def __init__(self, dim, batch_size):
  
    self.dim = dim
    self.batch_size = batch_size

    self.encoder_x = Encoder(name='GAN/encoder_x',activation=tf.nn.tanh, out_size=self.dim)
    self.decoder_x = Decoder(name='GAN/decoder_x',activation=tf.nn.tanh, out_size=self.dim)

    self.encoder_m = Encoder(name='GAN/encoder_m',activation=tf.nn.sigmoid, out_size=self.dim)
    self.decoder_m = Decoder(name='GAN/decoder_m',activation=tf.nn.sigmoid, out_size=self.dim)
    
    self.mask_generator = MaskGenerator()
    self.mask_discriminator = MaskDiscriminator()

    self.data_generator = DataGenerator()
    self.data_discriminator = DataDiscriminator()

    #self.adaptive_loss_x = AdaptiveLossFunction(self.dim, np.float32)
    #self.adaptive_loss_E_x = AdaptiveLossFunction(self.dim, np.float32)
    
    self.phase_train = tf.placeholder(tf.bool, name='phase_train')
    self.x = tf.placeholder(tf.float32, shape=[None, dim])
    self.M = tf.placeholder(tf.float32, shape = [None, dim])
    self.Z = tf.placeholder(tf.float32, shape = [None, dim])
    self.O = tf.placeholder(tf.float32, shape = [None, dim])
    self.E = tf.placeholder(tf.float32, shape = [None, dim])
    self.mu_x = tf.Variable(tf.zeros([dim], tf.float32))
    self.mu_E_x = tf.Variable(tf.zeros([dim], tf.float32))
    
    N_total = dim * batch_size
    N_observed = tf.reduce_sum(self.M)
    N_missing = dim * batch_size - N_observed

    if 1 == tf.cond(self.phase_train, lambda: tf.constant(1), lambda: tf.constant(0)):
      use_dropout = True
    else:
      use_dropout = False

    self.tau = 0.2
    self.alpha = 0.1
    self.beta = 0.1
    self.delta = 100
    self.lmbda = 1
    
    self.x_hat = self.M * self.x + (1 - self.M) * self.O
    self.x_enc = self.encoder_x(self.x_hat, use_dropout=False)
    self.x_enc_hat = self.encoder_x(self.x_hat, use_dropout=True)
    self.x_dec = self.decoder_x(self.x_enc_hat)
    
    self.m_enc = self.encoder_m(self.M, use_dropout=False)
    self.m_enc_hat = self.encoder_m(self.M, use_dropout=True)
    self.m_dec = self.decoder_m(self.m_enc_hat)
    
    self.G_x = self.data_generator(self.Z, self.m_enc)
    self.G_m = self.mask_generator(self.E, self.x_enc)
    
    self.X_fake = self.G_m * self.G_x + (1 - self.G_m) * self.tau
    self.X_real = self.M * self.x + (1 - self.M) * self.tau
    
    self.D_m_real = self.mask_discriminator(self.M, self.x_enc)
    self.D_m_fake = self.mask_discriminator(self.G_m, self.x_enc)

    self.D_x_real = self.data_discriminator(self.X_real, self.m_enc)
    self.D_x_fake = self.data_discriminator(self.X_fake, self.m_enc)

    self.M_loss = tf.reduce_mean(self.D_m_real) - tf.reduce_mean(self.D_m_fake)
    self.G_M_loss = - tf.reduce_mean(self.D_m_fake)
    
    self.X_loss = tf.reduce_mean(self.D_x_real) - tf.reduce_mean(self.D_x_fake)
    
    self.G_X_loss = - tf.reduce_mean(self.D_x_fake)
    
    self.abs_res_x = tf.math.abs(self.M * self.x - self.M * self.G_x)
    self.abs_res_E_x = tf.math.abs(self.M * self.x - self.M * self.x_dec)
    self.abs_res_m = tf.math.abs(self.M - self.G_m)
    
    self.MAE_loss_x = tf.reduce_mean(self.abs_res_x)
    self.MSE_loss_m = tf.reduce_mean(self.abs_res_m**2)
    
    self.log_res_x = tf.math.log(self.M * self.x + 1 + 1e-7) - tf.math.log(self.M * self.G_x + 1 + 1e-7)
    self.log_res_E_x = tf.math.log(self.M * self.x + 1 + 1e-7) - tf.math.log(self.M * self.x_dec + 1 + 1e-7)
    
    self.adp_loss_x = self.abs_res_x #tf.reduce_mean(self.adaptive_loss_x(self.abs_res_x - self.mu_x[tf.newaxis, :]))
    self.adp_loss_E_x = self.abs_res_E_x # tf.reduce_mean(self.adaptive_loss_E_x(self.abs_res_E_x - self.mu_E_x[tf.newaxis, :]))
    
    self.recon_loss_x = tf.reduce_mean(tf.math.abs(self.log_res_x))
    self.recon_loss_E_x = tf.reduce_mean(tf.math.abs(self.log_res_E_x))
    self.recon_loss_m = tf.reduce_mean(tf.keras.backend.binary_crossentropy(self.M, self.G_m))
    
    # min(L_e)
    self.E_x_loss = self.recon_loss_E_x #tf.reduce_mean((self.M * self.x  - self.M * self.x_dec)**2)
    self.E_m_loss = tf.reduce_mean((self.M  - self.m_dec)**2)
    
    # min(G_x)
    self.G_x_loss = self.G_X_loss + self.recon_loss_x
    # max(D_x)
    self.D_x_loss = -self.X_loss #+ self.gradient_penalty(self.x, self.m_enc, self.data_discriminator)
    
    # min(G_m)
    self.G_m_loss = self.G_M_loss + self.recon_loss_m + self.alpha * self.X_loss
    # max(D_m)
    self.D_m_loss = -self.M_loss #+ self.gradient_penalty(self.M, self.x_enc, self.mask_discriminator)
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
  
      all_vars = tf.global_variables()
      with tf.variable_scope('GAN/training', reuse=tf.AUTO_REUSE) as var_scope:
        
        #loss_vars_x = list(self.adaptive_loss_x.trainable_variables) + [self.mu_x]
        #loss_vars_E_x = list(self.adaptive_loss_E_x.trainable_variables) + [self.mu_E_x]

        self.E_m_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0002). \
          minimize(self.E_m_loss, var_list=(self.encoder_m.vars + self.decoder_m.vars))       
                          
        self.E_x_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0002). \
          minimize(self.E_x_loss, var_list=(self.encoder_x.vars + self.decoder_x.vars))
          
        self.G_m_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=0.002). \
          minimize(self.G_m_loss, var_list=self.mask_generator.vars)
        self.D_m_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001). \
          minimize(self.D_m_loss, var_list=self.mask_discriminator.vars)
  
        self.G_x_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001). \
          minimize(self.G_x_loss, var_list=(self.data_generator.vars))
        self.D_x_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00005). \
          minimize(self.D_x_loss, var_list=self.data_discriminator.vars)
  
  def gradient_penalty(self, real, cond, f):
    def interpolate(a):
        beta = tf.random_uniform(shape=tf.shape(a), minval=0., maxval=1.)
        _, variance = tf.nn.moments(a, list(range(a.shape.ndims)))
        b = a + 0.5 * tf.sqrt(variance) * beta
    
        shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
        alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
        inter = a + alpha * (b - a)
        inter.set_shape(a.get_shape().as_list())
    
        return inter
    
    x = interpolate(real)
    pred = f(x, cond)
    gradients = tf.gradients(pred, x)[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=list(range(1, x.shape.ndims))))
    gp = tf.reduce_mean((slopes - 1.)**2)
    return gp
  
  def train(self, sess, norm_data_x, data_m, sparse_data_x):
    
    no, dim = norm_data_x.shape
    
    self.n_epochs = 1300
    self.stopping_running_avg = 25
    self.stopping_threshold = 0.0001
    self.stopping_patience = 50
    self.stopping_step = 0
    self.stopping_run_in = 500
    self.enc_stop = 750
    self.loss_recording_interval = 20
    
    self.losses = {
      'E_x': [], 'E_m': [],
      'D_x': [], 'D_m': [], 
      'G_x': [], 'G_m': [], 
      'R_x': [], 'R_m': [], 
      'MAE_x': [], 'MSE_m': [] 
    }
    
    self.sess = sess
    self.sess.run(tf.global_variables_initializer())
    
    self.best_loss = None
    self.running_loss = []
    
    e_x_loss, e_m_loss, recon_x_loss = 0, 0, 0
    d_m_loss, g_m_loss, d_x_loss, g_x_loss = 0, 0, 0, 0
    recon_x_loss, recon_m_loss = 0, 0
    mae_loss_x, mse_loss_m = 0, 0
    loss_value, best_loss = None, None
    enc_periods, total_periods = 0, 0
    #self.plot_umap(sparse_data_x)
    
    for epoch in range(1, self.n_epochs+1):
    
      batch_idx = sample_batch_index(no, self.batch_size)
        
      X_mb = norm_data_x[batch_idx, :]  
      M_mb = data_m[batch_idx, :]
    
      Z_mb = uniform_sampler(-1, 1, self.batch_size, self.dim) 
      E_mb = uniform_sampler(-1, 1, self.batch_size, self.dim) 
      O_mb = uniform_sampler(-1, 1, self.batch_size, self.dim) 
    
      feed_dict = {
        self.x: X_mb, self.M: M_mb, 
        self.Z: Z_mb, self.E: E_mb, 
        self.O: O_mb, self.phase_train: True 
      }
      
      if epoch < self.enc_stop:
        _, e_x_loss, x_enc = sess.run([self.E_x_train_op, self.E_x_loss, self.x_enc], feed_dict = feed_dict)
        _, e_m_loss = sess.run([self.E_m_train_op, self.E_m_loss], feed_dict = feed_dict)
        
      if epoch > self.enc_stop:
        break
        _, d_m_loss = sess.run([self.D_m_train_op, self.D_m_loss], feed_dict = feed_dict)
        _, g_m_loss = sess.run([self.G_m_train_op, self.G_m_loss], feed_dict = feed_dict)
        
        _, d_x_loss = sess.run([self.D_x_train_op, self.D_x_loss], feed_dict = feed_dict)
        _, g_x_loss, recon_x_loss, recon_m_loss, mae_loss_x, mse_loss_m = \
          sess.run([self.G_x_train_op, self.G_x_loss, self.recon_loss_x, self.recon_loss_m,
          self.MAE_loss_x, self.MSE_loss_m], feed_dict = feed_dict)
      
      if epoch % self.loss_recording_interval == 0:
        self.losses['E_m'].append(e_m_loss)
        self.losses['E_x'].append(e_x_loss)
        self.losses['G_m'].append(g_m_loss)
        self.losses['D_m'].append(d_m_loss)
        self.losses['G_x'].append(g_x_loss)
        self.losses['D_x'].append(d_x_loss)
        self.losses['R_x'].append(recon_x_loss)
        self.losses['R_m'].append(recon_m_loss)
        self.losses['MAE_x'].append(mae_loss_x)
        self.losses['MSE_m'].append(mse_loss_m)
        
        if epoch < self.enc_stop: 
          enc_periods += 1
          total_periods += 1
        else: 
          total_periods += 1
      
      print(epoch, e_m_loss, e_x_loss, mse_loss_m, mae_loss_x)
      
      # Early stopping criterion
      # self.running_loss.append(mae_loss_x)
      # if len(self.running_loss) > self.stopping_running_avg: 
      #   self.running_loss.pop(0)
      # delta_x = (np.mean(self.running_loss) - mae_loss_x)
      # 
      # if epoch > self.stopping_run_in and delta_x < self.stopping_threshold:
      #   self.stopping_step += 1
      # else:
      #   self.stopping_step = 0
      #   
      # if self.stopping_step >= self.stopping_patience:
      #   print("Early stopping; ", np.mean(self.running_loss), delta_x, mae_loss_x)
        
    self.plot_losses(self.losses, enc_periods, total_periods)
        
    return self.impute(norm_data_x, data_m)
  
  def impute(self, norm_data_x, data_m):
    
    no, dim = norm_data_x.shape
    M_mb = data_m
    X_mb = norm_data_x
    
    Z_mb = uniform_sampler(-1, 1, no, dim) 
    E_mb = uniform_sampler(-1, 1, no, dim) 
    O_mb = uniform_sampler(-1, 1, no, dim) 
    
    x_enc, x_dec, G_x = self.sess.run([self.x_enc, self.x_dec, self.G_x], feed_dict = {
        self.x: X_mb, 
        self.M: M_mb, 
        self.Z: Z_mb, 
        self.E: E_mb,
        self.O: O_mb,
        self.phase_train: False
    })
      
    return M_mb * X_mb + (1-M_mb) * x_dec
    
  def plot_losses(self, losses, enc_stop, n_iter):
  
    plt.title('Autoencoder, data generator, and data critic losses over time')
    color_map = cm.rainbow(np.linspace(0,1,8))
    
    first_part = [20 * i for i in range(0, enc_stop)]
    second_part = [20 * i for i in range(enc_stop, n_iter)]
    
    plt.plot(first_part, losses['E_x'][0:enc_stop]/np.max(losses['E_x']), 
      label='Encoder loss (mask)', lw=2, alpha=0.8, c=color_map[0])
    plt.plot(first_part, losses['E_m'][0:enc_stop]/np.max(losses['E_m']), 
      label='Encoder loss (data)', lw=2, alpha=0.8, c=color_map[1])
    
    plt.plot(second_part, losses['G_m'][enc_stop:]/np.max(losses['G_m']), 
      label='Generator loss (mask)', lw=2, alpha=0.8, c=color_map[2])
    plt.plot(second_part, losses['D_m'][enc_stop:]/np.max(losses['D_m']), 
      label='Critic loss (mask)', lw=2, alpha=0.8, c=color_map[3])
      
    plt.plot(second_part, losses['G_x'][enc_stop:]/np.max(losses['G_x']), 
      label='Generator loss (data)',lw=2, alpha=0.8, c=color_map[4])
    plt.plot(second_part, losses['D_x'][enc_stop:]/np.max(losses['D_x']), 
      label='Critic loss (data)', lw=2, alpha=0.8, c=color_map[5])

    plt.plot(second_part, losses['MSE_m'][enc_stop:]/np.max(losses['MSE_m']), 
      label='Mean squared error (mask)', lw=2, alpha=0.8, c=color_map[6])
    plt.plot(second_part, losses['MAE_x'][enc_stop:]/np.max(losses['MAE_x']), 
      label='Mean absolute error (data)', lw=2, alpha=0.8, c=color_map[7])
    
    plt.xlabel('Number of training epochs',fontsize=6)
    plt.legend()
    
    plt.show()
  
  def plot_umap(self, X, colors=None):
    
    print('Fitting UMAP')
    low_embedding = UMAP(n_neighbors=4,min_dist=0.3, n_components=3).fit(X)
    X_r = low_embedding.transform(X)
    
    high_embedding = UMAP(n_neighbors=30,min_dist=0.0, n_components=16).fit(X)
    X_r_high = high_embedding.transform(X)
    
    if colors is None:
      print('Fitting clusters')
      clusters = HDBSCAN(min_samples=20, min_cluster_size=50).fit_predict(X_r_high)
      print(len(np.unique(clusters)))
      
      labels = [str(x) for x in clusters]
      n_clusters = len(np.unique(clusters)) 
      
      print(n_clusters)
      color_map = cm.rainbow(np.linspace(0,1,n_clusters))
      colors = [color_map[c] for c in clusters]
      self.cluster_colors = colors
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_r_high[:,0], X_r_high[:,1], X_r_high[:,2], c=colors, marker='o', s=1.5, alpha=0.8)
    
    plt.title("UMAP")
    plt.axis('tight')
    plt.show()
    
  