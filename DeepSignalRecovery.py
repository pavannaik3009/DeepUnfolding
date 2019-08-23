import tensorflow as tf
import numpy as np

seed = 3
tf.set_random_seed(seed)
np.random.seed(seed)

system_stat = 'Normal'

def q(x):
  eps = 0.0001
  y = 0.5*tf.erfc(x/np.sqrt(2))
  return y + eps


def grad_q(x):
  eps = 0.0001
  y = (-1./tf.sqrt(2.*np.pi)) * tf.exp(-0.5*(x**2))
  return y +eps

def grad2_q(x):
  eps = 0.0001
  y = tfe.gradients_function(q)
  z = y(x)
  z = tf.squeeze(z, 0)
  return z

def eta(x):
  eps = tf.constant(1)
  y = (grad_q(x))/(q(x))
  return y



def generate_data(B, H_, thr_, Ny, Nx, x_delta, thr_delta, sigma_delta, mode='IID'):
  x_true = np.zeros([B,Nx,1])
  H = np.zeros([B,Ny,Nx])
  thr = np.zeros([B,Ny,1])
  noise = np.zeros([B,Ny])
  C = np.zeros([B,Ny,Ny])
  for i in range(B):
    x_true[i,:,:] = np.random.uniform(low = x_delta[0], high=x_delta[1], size=(Nx,1))
    H[i,:,:] = H_
    thr[i,:] = thr_
    if mode=='IID':
      sigma = np.random.uniform(low=sigma_delta[0], high=sigma_delta[1])
      C[i,:,:] = (sigma**2)*np.eye(Ny)
      noise[i,:] = np.random.multivariate_normal(  np.zeros([Ny])  , C[i,:,:] )
    elif mode=='WNI':
      sigma = np.random.uniform(low=sigma_delta[0], high=sigma_delta[1], size=(Ny))
      C[i,:,:] = np.diag(sigma**2)
      noise[i,:] = np.random.multivariate_normal(  np.zeros([Ny])  , C [i,:,:])
  y = np.squeeze(np.matmul(H,x_true),-1) + noise
  r = np.sign(y - np.squeeze(thr,-1))
  return y, H, x_true, C, r, thr


def weight_variable(shape):
  initial = tf.round(tf.random_normal(shape, stddev=0.01))
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.random_normal(shape, stddev=0.01)
  return tf.Variable(initial)



Ny = 50
Nx = 2
B= 500
L= 15 

model_name= 'Ny'+str(Ny)+'Nx'+str(Nx)+'B'+str(B)+'L'+str(L)
x_delta = [3, 7]
thr_delta = [-20,20]
sigma_delta = [0.1, 0.5] 
mode='WNI'

H1 = np.random.randn(Ny,Nx)
thr1 = np.random.uniform(low = thr_delta[0], high = thr_delta[1], size=(Ny,1))
np.savetxt('H1'+model_name+'.csv', H1, delimiter=",")
np.savetxt('thr1'+model_name+'.csv', thr1, delimiter=",")

startingLearningRate = 0.0001
decay_factor = 0.97
decay_step_size = 1000
train_iter = 20000
res_alpha=0.9


sess = tf.InteractiveSession()

org_signal = tf.placeholder(tf.float32, shape=[None,Nx,1], name='org_siganl')
r = tf.placeholder(tf.float32, shape=[None,Ny], name= 'one-bit-data')
C = tf.placeholder(tf.float32, shape = [None,Ny, Ny], name = 'noise-cov')
tau = tf.placeholder(tf.float32, shape=[None,Ny,1], name='Thresholds')
H = tf.placeholder(tf.float32, shape=[None,Ny,Nx], name='Sensing-Matrix')
delta = tf.Variable(1.)
Omega = tf.linalg.diag(r)
C_ = tf.linalg.inv(tf.sqrt(C))
Omega_tilde = tf.matmul(Omega,C_)
H_tilde = tf.matmul(Omega_tilde,H)
H_tilde_T = tf.transpose(H_tilde)
H_tilde_T = tf.transpose(H_tilde_T, perm=[2,0,1])
tau_tilde = tf.matmul(Omega_tilde, tau)
tau_tilde = tf.squeeze(tau_tilde,-1)
X = [] 
X.append(tf.zeros([B,Nx])) 

loss=[]
loss.append(tf.zeros([]))


for i in range(1,L): 
  A = weight_variable([Ny,Ny])
  BB = weight_variable([Ny,Ny])
  W1 = weight_variable([Nx,Nx])
  W2 = weight_variable([Nx,Nx])
  b1 =  bias_variable([Nx,1])
  X_ = tf.matmul(H_tilde,tf.expand_dims(X[-1],-1))
  b = bias_variable([Ny,1])
  temp1 = tf.matmul(tau_tilde,A)
  temp2 = tf.matmul(tf.squeeze(X_,-1),BB)
  
  temp3 = eta(tf.expand_dims(temp1 - temp2,-1) + b)
  temp4 = tf.matmul(H_tilde_T, temp3)
  WW1 = tf.matmul(tf.squeeze(temp4,-1),W1)
  WW2 = tf.matmul(X[-1],W2)
  temp5 = tf.nn.relu(WW2 - WW1 + tf.squeeze(b1,-1))
  X.append(temp5)
  X[i] = (1-res_alpha)*X[i]+res_alpha*X[i-1]
 
   

LOSS = tf.reduce_mean(tf.square(tf.squeeze(org_signal,-1) - X[-1]))
X_HAT = X[-1]


global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(startingLearningRate, global_step, decay_step_size, decay_factor, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(LOSS)
init_op=tf.global_variables_initializer()
saver = tf.train.Saver()
sess.run(init_op)


for i in range(300):
  y1, _H1, x1 , C1, r1, taus = generate_data(B, H1, thr1, Ny, Nx, x_delta, thr_delta, sigma_delta, mode)
  train_step.run(feed_dict={org_signal: x1, 
                            r: r1,
                            C: C1,
                            tau: taus,
                            H: _H1
                           
                           })
  if i%100==0:
    train_accuracy = LOSS.eval(feed_dict = {org_signal: x1, 
                           r: r1,
                           C: C1,
                           tau: taus,
                           H: _H1
                           
                           })
    
    estimator = X_HAT.eval(feed_dict = {org_signal: x1, 
                           r: r1,
                           C: C1,
                           tau: taus,
                           H: _H1
                           
                           })
    
    print (i, train_accuracy) 
    print ('#######')
    print (x1[0,:,:].T)
    print (estimator[0,:])
saver.save(sess, '/content/DeepUnfolding/'+model_name)

import time as tm
from google.colab import files 
test_iter = 50

MSE = np.zeros([test_iter])
NMSE = np.zeros([test_iter])
TIME = np.zeros([test_iter])

for i in range(test_iter):
  y1, _H1, x1 , C1, r1, taus = generate_data(B, H1, thr1, Ny, Nx, x_delta, thr_delta, sigma_delta, mode)
  feeed = {org_signal: x1, r: r1, C: C1, tau: taus, H: _H1}
  tic = tm.time()
  MSE[i] = LOSS.eval(feed_dict = feeed)
  toc = tm.time()
  NMSE[i] = MSE[i]/np.mean(np.squeeze(x1,-1)**2)
  TIME[i] = toc - tic
  print('Iteration: %i - MSE: %f - NMSE: %f - TIME: %f' % (i,MSE[i], NMSE[i], TIME[i]))
TIME_AVG = np.mean(TIME)/B
MSE_AVG = np.mean(MSE)
NMSE_AVG = np.mean(NMSE)
print ('-------------Final Results-------------')
print ('MSE: %f - NMSE: %f - TIME: %f per symbol' % (MSE_AVG, NMSE_AVG, TIME_AVG))
print ('---------------------------------------')
np.savez('/content/'+model_name+'.pavan', TIME=TIME, TIME_AVG=TIME_AVG, MSE=MSE, MSE_AVG=MSE_AVG, NMSE=NMSE, NMSE_AVG=NMSE_AVG, Ny=Ny, B=B, thr1=thr1, H1=H1, Nx=Nx, L=L,test_iter=test_iter)
