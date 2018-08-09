import tensorflow as tf
import numpy as np
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# from TTGP.covariance import SE
from TTGP.covariance import SE_multidim
from TTGP.projectors import FeatureTransformer, LinearProjector, Identity
from TTGP.gpc_runner import GPCRunner

N_samples = 100
N_dim = 2
N_inducingpoints=20
noise = 0.1
X,y = make_classification(n_samples=N_samples,n_features=N_dim,n_classes=2,n_clusters_per_class=2,n_informative=N_dim,n_redundant=0)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)
batch_size=40
n_epochs=100

with tf.Graph().as_default():
    data_dir = "data_class/"
    n_inputs = 10
    mu_ranks = 10
    projector = Identity(D=N_dim)
    C = 2
    cov = SE_multidim(C,0.7, 3.5, 0.1,projector)
    # cov = SE(0.7, 0.2, 0.1,projector)
    lr = 1e-2
    log_dir = 'log'
    save_dir = 'models/proj_nn_4.ckpt'
    model_dir = save_dir
    load_model = False#True
    runner=GPCRunner(n_inputs, mu_ranks, cov, X=X,X_test=X_test,y=y.reshape(-1,1),y_test=y.reshape(-1,1),
                lr=lr, n_epoch=n_epochs, batch_size=batch_size)
    runner.run_experiment()
