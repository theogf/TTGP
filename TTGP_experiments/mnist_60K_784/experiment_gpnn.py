import tensorflow as tf
import numpy as np
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# from TTGP.covariance import SE
from TTGP.covariance import SE_multidim
from TTGP.projectors import FeatureTransformer, LinearProjector, Identity
from TTGP.gpc_runner import GPCRunner

N_samples = 1000
N_dim = 20
N_inducingpoints=2
noise = 0.1
X,y = make_classification(n_samples=N_samples,n_features=N_dim,n_classes=2,n_clusters_per_class=2,n_informative=N_dim,n_redundant=0)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)
batch_size=100
n_epochs=111
y_p,m = 0,0
with tf.Graph().as_default():
    data_dir = "data_class/"
    n_inputs = 30
    mu_ranks = 10
    projector = Identity(D=N_dim)
    C = 2
    cov = SE_multidim(C,0.7, 1.0, 0.1,projector)
    # cov = SE(0.7, 0.2, 0.1,projector)
    lr = 1e-2
    runner=GPCRunner(n_inputs, mu_ranks, cov, X=X,X_test=X_test,y=y.reshape(-1,1),y_test=y_test,
                lr=lr, n_epoch=n_epochs, batch_size=batch_size,batch_test=False)
    y_p,m = runner.run_experiment()
