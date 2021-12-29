'''
title           : BlockchainbasedFederatedLearning.py
description     : Towards an Efficient IoMT: Privacy Infrastructure for COVID-19 Pandemic based on Federated Learning and Blockchain Technology
authors         : Omaji Samuel, Akogwu Blessing Omojo, Abdulkarin Musa Onuja, Yunisa Sunday, Prayag Tiwari, Deepak
                  Gupta, Ghulam Hafeez, Adamu Sani Yahaya, Oluwaseun Jumoke Fatoba, and Shahab Shamshirband
date_created    : 20211112
date_modified   : Not Applicable
version         : 0.1
usage           : python FedMedChain.py
                  python FedMedChain.py -p 5000
                  python FedMedChain.py --port 5000
python_version  : 3.7.9
Comments        : The proposed model is a modification of [1] and [2]. The proposed blockchain-based federated learning (FL) model uses COVID-19 epidemiological data from different centres for disease control (CDCs) to predict the progression of COVID-19 infectious and susceptible individuals.
                  Our target variable is a continuous Quantitative measure of the COVID-19 infectious and susceptible individuals' progression, which is solved by linear regression. Moreover, a deep learning method, such as the convolutional neural network (CNN) method can be adopted.
 
                  We consider a horizontally federated learning model in which COVID-19 data having the same features is divided into more than one CDCs for training the model
                  The aim of this research is to consider the entire training set for improving the federated model that is trained locally at each CDC 
                  By generalization, this study aims that the records of COVID-19 patients must never leave CDC whether it is encrypted or not. Besides, the records cannot 
                  be shared unless they are encrypted and neither the FL server nor patients should infer the sources of the records. We deploy differential privacy at the beginning of the protocol to solve the privacy concerns of patients. 
                  [1]: http://scikit-learn.org/stable/datasets/index.html#diabetes-dataset
                  [2]: https://research.googleblog.com/2017/04/federated-learning-collaborative.html

'''
from __future__ import division
import time
from contextlib import contextmanager
import numpy as np
import time
from contextlib import contextmanager
import numpy as np
from sklearn.datasets import load_diabetes

import phe as paillier
# import library for plotting graph
import matplotlib.pyplot as plt 
from cProfile import label
from sklearn.metrics import mean_absolute_error
# create library for custering method
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans 
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from numpy import array
from CovidBlockchainEpidemic import  SESIAISQEQHR
import random


seed = 43 # pseudo random number generation
np.random.seed(seed)
mse2=[] # mean square error
mse1=[] # mean square error
pred1=[] # prediction results for blockchain based federated server
pred2=[] # prediction results for local server

# 
def user_choice(r,c,a,b,e):
    """ b is the CDC that valued his reward against privacy; a is CDC that valued is privacy against its 
        reward; e is the probability of breach (0< e<=1); c is the cost of privacy; r is the reward; and N is the choice of a CDC """
    return b *r + a * e * c

def aggregator_total_cost(k,r,N):   
    """ Calculates the aggregator's total cost. Note that the aggregator is the leader of the network, who decides the reward that each node
    in the blockchain will get """
    TC_ag=[] # total cost of aggregator
    for c in range(N):
        TC_ag.append(k + r *c)
    return  TC_ag
    
def aggregator_revenue(r,N):
    """ Calculates the aggregator's revenue, which indicates the profit achieved for sharing COVID-19 data"""
    R_ag=[] # revenue of aggregator
    for c in range(N):
        R_ag.append(r * c)
    return R_ag

def aggregator_utility(R,r,k,N):
    """ Calculates the aggregator's utility """
    U_ag=[] # utility of aggregator
    for c in range(N):
        U_ag.append((R-r)*c + k)
    return U_ag

def CDC_revenue(N, accuracy, data_worth):
    """ Calculates the CDC nodes' revenue """
    R_CDC=[] # revenue for each CDC
    for c in range(N):
        R_CDC.append(data_worth* accuracy*c)
    return R_CDC

def CDC_utility(R,N):
    """ Calculates the CDC nodes' utility"""
    U_CDC =[] # utility of each CDC
    for c in range(N):
        U_CDC.append(R_worker[c] - R)
    return U_CDC


def get_data(n_CDCs):
    """
    Get COVID-19 data from CDCs and split them into train/test.
    Return training, target lists for `n_CDCs` and a holdout test set
    """
    print("Loading data")
    modelSESIAISQEQHR = SESIAISQEQHR()
    dataset = modelSESIAISQEQHR.model([150, 100, 100, 50, 20, 1, 1, 2, 16],
             [0, 4420],
             100000,
             {'phiB': 0.08 ,
               'Rq': 0.7, 
               'PT': 0.8,
               'sigma': 0.2,
               'lambdaB': 0.5,
                'Ca': 0.2, 
                'Cs': 0.1,
                'phis': 0.3, 
                'taus': 0.3,
                'taua': 0.201,
                'tauh': 0.130,
                'deltas': 0.11,
                'deltaa': 0.102,
                'deltaq': 0.2,
                'Cm': 0.2,
                'deltah': 0.003,
                'muB': 0.3,
                'etaB': 0.735,
                'r': 0.13,
                'Cm1': 0.5,
                'Cq1': 0.3,
                'Bc': 10,
                'Ba': 100,
                'Hr': 1414
            })
    print('Epidemiological dataset:', dataset)
    X = dataset['SQ']
    X = np.reshape(X, [442, 10])
    y = np.zeros([442, ])
    for i in range(len(y)):
        y[i] = random.randrange(1,200)

    print('Training Set: ', X.shape)
    print('Testing Set: ', y.shape)
    # Add constant to emulate intercept
    X = np.c_[X, np.ones(X.shape[0])]

    # The features are already preprocessed
    # Shuffle
    perm = np.random.permutation(X.shape[0])
    X, y = X[perm, :], y[perm]

    # Select test at random
    test_size = 50
    test_idx = np.random.choice(X.shape[0], size=test_size, replace=False)
    train_idx = np.ones(X.shape[0], dtype=bool)
    train_idx[test_idx] = False
    X_test, y_test = X[test_idx, :], y[test_idx]
    X_train, y_train = X[train_idx, :], y[train_idx]

    # Split train among multiple CDCs.
    # The selection is not at random. We simulate the fact that each CDC
    # sees a potentially very different sample of patients.
    X, y = [], []
    step = int(X_train.shape[0] / n_CDCs)
    for c in range(n_CDCs):
        X.append(X_train[step * c: step * (c + 1), :])
        y.append(y_train[step * c: step * (c + 1)])

    return X, y, X_test, y_test

# a function for the MSE
def mean_square_error(y_pred, y):
    """ 1/m * \sum_{i=1..m} (y_pred_i - y_i)^2 """
    return np.mean((y - y_pred) ** 2)

# a function for encryption
def encrypt_vector(public_key, x):
    return [public_key.encrypt(i) for i in x]

# a function for decryption
def decrypt_vector(private_key, x):
    return np.array([private_key.decrypt(i) for i in x])

# a function for additive homomorphic encryption
def sum_encrypted_vectors(x, y):
    if len(x) != len(y):
        raise ValueError('Encrypted vectors must have the same size')
    return [x[i] + y[i] for i in range(len(x))]

@contextmanager
def timer():
    """Helper for measuring runtime"""

    time0 = time.perf_counter()
    yield
    print('[elapsed time: %.2f s]' % (time.perf_counter() - time0))

# a function for MAPE
def Mean_absolute_percentage_error(y_true,y_pred):
    """ Calculate the mean absolute percentage error of the true y_true and predicted y_pred values"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class Server:
    """Private key holder. Decrypts the average gradient"""

    def __init__(self, key_length):
         keypair = paillier.generate_paillier_keypair(n_length=key_length)
         self.pubkey, self.privkey = keypair

    def decrypt_aggregate(self, input_model, n_CDCs):
        return decrypt_vector(self.privkey, input_model) / n_CDCs


class CDC:
    """Runs linear regression with local data or by gradient steps,
    where gradient can be passed in.

    Using public key can encrypt locally computed gradients.
    """

    def __init__(self, name, X, y, pubkey):
        self.name = name
        self.pubkey = pubkey
        self.X, self.y = X, y
        self.weights = np.zeros(X.shape[1])

    def fit(self, n_iter, eta=0.01):
        """Linear regression for n_iter"""
        for _ in range(n_iter):
            gradient = self.compute_gradient()
            self.gradient_step(gradient, eta)

    def gradient_step(self, gradient, eta=0.01):
        """Update the model with the given gradient"""
        self.weights -= eta * gradient

    def compute_gradient(self):
        """Compute the gradient of the current model using the training set
        """
        delta = self.predict(self.X) - self.y
        return delta.dot(self.X) / len(self.X)

    def predict(self, X):
        """Score test data"""
        return X.dot(self.weights)

    def encrypted_gradient(self, sum_to=None):
        """Compute and encrypt gradient.

        When `sum_to` is given, sum the encrypted gradient to it, assumed
        to be another vector of the same size
        """
        gradient = self.compute_gradient()
        encrypted_gradient = encrypt_vector(self.pubkey, gradient)

        if sum_to is not None:
            return sum_encrypted_vectors(sum_to, encrypted_gradient)
        else:
            return encrypted_gradient


def blockchain_based_federated_learning(X, y, X_test, y_test, config):
    n_CDCs = config['n_CDCs']
    n_iter = config['n_iter']

    names = ['CDC {}'.format(i) for i in range(1, n_CDCs + 1)]

    # Instantiate the server and generate private and public keys
    # NOTE: using smaller keys sizes wouldn't be cryptographically safe
    server = Server(key_length=config['key_length'])

    # Instantiate the CDCs.
    # Each CDC gets the public key at creation and its own local dataset
    CDCs = []
    for i in range(n_CDCs):
        CDCs.append(CDC(names[i], X[i], y[i], server.pubkey))

    # The federated learning with gradient descent
    print('Running distributed gradient aggregation for {:d} iterations'
          .format(n_iter))
    for i in range(n_iter):

        # Compute gradients, encrypt and aggregate
        encrypt_aggr = CDCs[0].encrypted_gradient(sum_to=None)
        for c in CDCs[1:]:
            encrypt_aggr = c.encrypted_gradient(sum_to=encrypt_aggr)

        # Send aggregate to server and decrypt it
        aggr = server.decrypt_aggregate(encrypt_aggr, n_CDCs)

        # Take gradient steps
        for c in CDCs:
            c.gradient_step(aggr, config['eta'])

    print('Error (MSE) that each CDC gets after running the protocol:')
    for c in CDCs:
        y_pred = c.predict(X_test)
        pred1=y_pred
        #mse = mean_square_error(y_pred, y_test)
        #mse=mean_absolute_error(y_test,y_pred)
        mse=Mean_absolute_percentage_error(y_test,y_pred)
        mse1.append(mse)
        print('{:s}:\t{:.2f}'.format(c.name, mse))
    # Clustering
    X=array(y_pred)
    #X=X.reshape(4290,1) # convert to one dimension
    X=X.reshape(25,2) # convert to two dimension
    Clustering(X)
    
def local_learning(X, y, X_test, y_test, config):
    n_CDCs = config['n_CDCs']
    names = ['CDC {}'.format(i) for i in range(1, n_CDCs + 1)]

    # Instantiate the CDCs.
    # Each CDC gets the public key at creation and its own local dataset
    CDCs = []
    for i in range(n_CDCs):
        CDCs.append(CDC(names[i], X[i], y[i], None))

    # Each CDC trains a linear regressor on its own data
    print('Error (MSE) that each CDC gets on test set by '
          'training only on own local data:')
    for c in CDCs:
        c.fit(config['n_iter'], config['eta'])
        y_pred = c.predict(X_test)
        pred2=y_pred
        #mse = mean_square_error(y_pred, y_test)
        #mse=mean_absolute_error(y_test,y_pred)
        mse=Mean_absolute_percentage_error(y_test,y_pred)
        mse2.append(mse)
        print('{:s}:\t{:.2f}'.format(c.name, mse))
        # perform clustering
    X=array(y_pred)
    #X=X.reshape(4290,1) # convert to one dimension
    X=X.reshape(25,2) # convert to two dimension
    Clustering(X)

def Clustering(X):
    # choice any clustering method of your choice
    clust = OPTICS(min_samples=3, xi=.05, min_cluster_size=.05)
    # Run the fit
    clust.fit(X)

    labels_050 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=0.5)
    labels_200 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=2)

    space = np.arange(len(X))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 3)
    ax1 = plt.subplot(G[0, :])
    ax2 = plt.subplot(G[1, 0])
    ax3 = plt.subplot(G[1, 1])
    ax4 = plt.subplot(G[1, 2])

    # Reachability plot
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        ax1.plot(Xk, Rk, color, alpha=0.3)
    ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
    ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
    ax1.set_ylabel('Reachability (epsilon distance)')
    ax1.set_title('Reachability Plot')

    # OPTICS
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = X[clust.labels_ == klass]
        ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
    ax2.set_title('Automatic Clustering\nOPTICS')

    # DBSCAN at 0.5
    colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
    for klass, color in zip(range(0, 6), colors):
        Xk = X[labels_050 == klass]
        ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
    ax3.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], 'k+', alpha=0.1)
    ax3.set_title('Clustering at 0.5 epsilon cut\nDBSCAN')

    # DBSCAN at 2.
    colors = ['g.', 'm.', 'y.', 'c.']
    for klass, color in zip(range(0, 4), colors):
        Xk = X[labels_200 == klass]
        ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    ax4.plot(X[labels_200 == -1, 0], X[labels_200 == -1, 1], 'k+', alpha=0.1)
    ax4.set_title('Clustering at 2.0 epsilon cut\nDBSCAN')

    plt.tight_layout()
    plt.show()

# key_length=1024, eta=1.5;
if __name__ == '__main__':
    config = {
        'n_CDCs': 5,
        'key_length': 1024,
        'n_iter': 50,
        'eta': 0.01,
    }
    # load data, train/test split and split training data between CDCs
    X, y, X_test, y_test = get_data(n_CDCs=config['n_CDCs'])
    # first each CDC learns a model on its respective dataset for comparison.
    with timer() as t:
         local_learning(X, y, X_test, y_test, config)
    # and now the full glory of federated learning
    with timer() as t:
        blockchain_based_federated_learning(X, y, X_test, y_test, config)
        r=3
        R=5
        c= 0.4
        a=6
        b=0.6
        e=0.1
        k=0.1
        N=round(user_choice(r,c,a,b,e))
        print(N)
        TC=aggregator_total_cost(k,r,N) 
        print(TC)
        R_agg=aggregator_revenue(r,N) 
        print(R_agg)
        U_agg=aggregator_utility(R,r,k,N)
        print(U_agg)
        
    
    plt.figure(figsize=(10, 7))
    y_pos= np.arange(len(TC))
    performance=TC
    Objects=range(N)
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, Objects)
    plt.ylabel('Total Cost')
    plt.xlabel('Number of Nodes')
    plt.show()
    # plots the graphs for federated learning and local learning
    # plotting the points  
    
    x1=[1,2,3,4,5]
    plt.plot(x1, mse1,label='federated learning',color='green', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='blue', markersize=12); 
    
    plt.plot(x1, mse2,label='local learning',color='blue', linestyle='dashed', linewidth = 3, 
        marker='o', markerfacecolor='red', markersize=12) 
  
    # naming the x axis 
    plt.xlabel('Homes') 
    # naming the y axis 
    plt.ylabel('Error') 
  
    # giving a title to my graph 
    plt.title('federated learning versus local learning') 
    # show a legend on the plot 
    plt.legend() 
  
    # function to show the plot 
    plt.show() 
    
    
