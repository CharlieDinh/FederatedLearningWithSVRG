# Generate 100 users with heterogeneous data sizes.
# Perform normalization on each user's data (after data has been split)

from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from tqdm import trange
import numpy as np
import random
import json
import os

random.seed(1)
np.random.seed(1)
NUM_USERS = 100  
NUM_LABELS = 3

# Setup directory for train/test data
train_path = './data/train/mnist_train.json'
test_path = './data/test/mnist_test.json'

dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Get MNIST data
mnist = fetch_mldata('MNIST original', data_home='./data')
mnist_data = []

# Split data into each class
for i in trange(10):
    idx = mnist.target==i
    mnist_data.append(mnist.data[idx])

# Create user data split
# Assign 100 samples to each user
X = [[] for _ in range(NUM_USERS)]
y = [[] for _ in range(NUM_USERS)]
idx = np.zeros(10, dtype=np.int64)
for user in range(NUM_USERS):
    for j in range(NUM_LABELS):  # 3 labels for each users
        l = (user + j) % 10
        X[user] += mnist_data[l][idx[l]:idx[l]+10].tolist()
        y[user] += (l*np.ones(10)).tolist()
        idx[l] += 10


# Assign remaining sample by power law
user = 0
props = np.random.lognormal(
    0, 2., (10, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
props = np.array([[[len(v)-1000]] for v in mnist_data]) * \
    props/np.sum(props, (1, 2), keepdims=True)
for user in trange(NUM_USERS):
    for j in range(NUM_LABELS):  # 4 labels for each users
        l = (user + j) % 10
        num_samples = int(props[l, user//int(NUM_USERS/10), j])
        numran1 = random.randint(10, 200)
        numran2 = random.randint(1, 10)
        num_samples = (num_samples) * numran2 + numran1
        if(NUM_USERS <= 20):
            num_samples = num_samples * 2
        if idx[l] + num_samples < len(mnist_data[l]):
            X[user] += mnist_data[l][idx[l]:idx[l]+num_samples].tolist()
            y[user] += (l*np.ones(num_samples)).tolist()
            idx[l] += num_samples


train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

for i in range(NUM_USERS):
    print(i)
    uname = 'f_{0:05d}'.format(i)
    
    combined = list(zip(X[i], y[i]))
    random.shuffle(combined)
    X[i][:], y[i][:] = zip(*combined)
    num_samples = len(X[i])
    train_len = int(0.75*num_samples)
    test_len = num_samples - train_len
    
    # Create train data for each user, and scale it
    train_data['users'].append(uname)
    train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
    scaler = StandardScaler()
    train_data['user_data'][uname]['x'] = scaler.fit_transform(train_data['user_data'][uname]['x']).tolist()
    train_data['num_samples'].append(train_len)

    # Create test data for each user, and scale it
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
    test_data['user_data'][uname]['x'] = scaler.transform(test_data['user_data'][uname]['x']).tolist()
    test_data['num_samples'].append(test_len)


# print("Num_samples:", train_data['num_samples'])
# print("Total_samples:",sum(train_data['num_samples']))
with open(train_path,'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)
