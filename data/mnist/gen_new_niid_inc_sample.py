from sklearn.datasets import fetch_mldata
from tqdm import trange
import numpy as np
import random
import json
import os

random.seed(1)
np.random.seed(1)

NUMBER_USER = 100
NUMBER_ClASS_PER_USERS = 3
# Setup directory for train/test data
train_path = './data/train/mnist_niid_equal_train.json'
test_path = './data/test/mnist_niid_equal_train.json'


dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Get MNIST data, normalize, and divide by level
mnist = fetch_mldata('MNIST original', data_home='./data')
mu = np.mean(mnist.data.astype(np.float32), 0)
sigma = np.std(mnist.data.astype(np.float32), 0)
mnist.data = (mnist.data.astype(np.float32) - mu)/(sigma+0.001)

# set mean
num_sample_mean = len(mnist.data)//NUMBER_USER//NUMBER_ClASS_PER_USERS

#normal_std = np.sqrt(np.log(1 + (10/num_sample_mean)**2))
#normal_mean = np.log(num_sample_mean) - 10**2 / 2

#props = np.random.lognormal(100, 0,(NUMBER_USER, NUMBER_ClASS_PER_USERS))

mnist_data = []
for i in trange(10):
    idx = mnist.target == i
    mnist_data.append(mnist.data[idx])
print("Len of each class in mnist_data")
print([len(v) for v in mnist_data])

###### CREATE USER DATA SPLIT #######
# Assign 10 samples to each user (5 samples for each 2 type of labels for each user)
X = [[] for _ in range(NUMBER_USER)]
y = [[] for _ in range(NUMBER_USER)]
idx = np.zeros(10, dtype=np.int64)

for user in range(NUMBER_USER):
    for j in range(NUMBER_ClASS_PER_USERS):  # 2 labels
        l = (user+j) % 10
        # print("L:", l)
        X[user] += mnist_data[l][idx[l]:idx[l]+num_sample_mean].tolist()
        y[user] += (l*np.ones(num_sample_mean)).tolist()
        idx[l] += num_sample_mean
print("IDX1:", idx)  # counting samples for each labels

num_samples = num_sample_mean
# 200 sample for each class
for user in trange(NUMBER_USER):
    for j in range(NUMBER_ClASS_PER_USERS):
        l = (user+j) % 10
        if idx[l] + num_samples < len(mnist_data[l]):
            X[user] += mnist_data[l][idx[l]:idx[l]+num_samples].tolist()
            y[user] += (l*np.ones(num_samples)).tolist()
            idx[l] += num_samples
            # Assign remaining sample by power law

print("IDX2:", idx)  # counting samples for each labels

# Create data structure
train_data = {'users': [], 'user_data': {}, 'num_samples': []}
test_data = {'users': [], 'user_data': {}, 'num_samples': []}

# Setup 1000 users
for i in trange(NUMBER_USER, ncols=120):
    uname = 'f_{0:05d}'.format(i)

    combined = list(zip(X[i], y[i]))
    random.shuffle(combined)
    X[i][:], y[i][:] = zip(*combined)
    num_samples = len(X[i])
    train_len = int(0.9*num_samples)
    test_len = num_samples - train_len

    train_data['users'].append(uname)
    train_data['user_data'][uname] = {
        'x': X[i][:train_len], 'y': y[i][:train_len]}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {
        'x': X[i][train_len:], 'y': y[i][train_len:]}
    test_data['num_samples'].append(test_len)

print("Num_samples:", train_data['num_samples'])
print("Total_samples:", sum(train_data['num_samples']))

with open(train_path, 'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)

print("Finish Generating Samples")
