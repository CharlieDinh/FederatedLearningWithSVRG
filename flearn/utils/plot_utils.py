import matplotlib.pyplot as plt
import h5py
import numpy as np

def simple_read_data(loc_ep, alg):
    hf = h5py.File("./results/"+'{}_{}.h5'.format(alg, loc_ep), 'r')
    rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
    rs_train_acc = np.array(hf.get('rs_train_acc')[:])
    rs_train_loss = np.array(hf.get('rs_train_loss')[:])
    return rs_train_acc, rs_train_loss, rs_glob_acc

def plot_summary_one_figure(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], algorithms_list=[], batch_size=0, dataset = ""):
    
    Numb_Algs = len(algorithms_list)
    train_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))
    glob_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    algs_lbl = algorithms_list.copy()
    for i in range(Numb_Algs):
        if(lamb[i] > 0):
            algorithms_list[i] = algorithms_list[i] + "_prox_" + str(lamb[i])
            algs_lbl[i] = algs_lbl[i] + "_prox"
        algorithms_list[i] = algorithms_list[i] + \
            "_" + str(learning_rate[i]) + "_" + str(num_users) + \
            "u" + "_" + str(batch_size[i]) + "b"
        train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(
            simple_read_data(loc_ep1[i], dataset + algorithms_list[i]))[:, :Numb_Glob_Iters]
        algs_lbl[i] = algs_lbl[i]
    
    plt.figure(1)
    MIN = train_loss.min() - 0.001
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    for i in range(Numb_Algs):
        plt.plot(train_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + str(lamb[i])+ "_"+str(loc_ep1[i])+"e" + "_" + str(batch_size[i]) + "b")
    plt.legend(loc='lower right')
    plt.ylabel('Training Accuracy')
    plt.xlabel('Global rounds ' + '$K_g$')
    plt.title(dataset.upper())
    #plt.ylim([0.8, glob_acc.max()])
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'train_acc.png')
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_acc.pdf')

    plt.figure(2)
    for i in range(Numb_Algs):
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + str(lamb[i])+
                 "_"+str(loc_ep1[i])+"e" + "_" + str(batch_size[i]) + "b")
        #plt.plot(train_loss1[i, 1:], label=algs_lbl1[i])
    plt.legend(loc='upper right')
    plt.ylim([MIN, MIN+ 0.3])
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    plt.title(dataset.upper())
    #plt.ylim([train_loss.min(), 1])
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'train_loss.png')
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.pdf')

    plt.figure(3)
    for i in range(Numb_Algs):
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i]+str(lamb[i])+"_"+str(loc_ep1[i])+"e" + "_" + str(batch_size[i]) + "b")
        #plt.plot(glob_acc1[i, 1:], label=algs_lbl1[i])
    plt.legend(loc='lower right')
    #plt.ylim([0.6, glob_acc.max()])
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds ')
    plt.title(dataset.upper())
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'glob_acc.png')
    #plt.savefig(dataset + str(loc_ep1[1]) + 'glob_acc.pdf')

def plot_summary_two_figures(num_users=100, loc_ep1=[], Numb_Glob_Iters=10, lamb=[], learning_rate=[], algorithms_list=[], batch_size = 0, dataset = ""):
    Numb_Algs = len(algorithms_list)
    train_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))
    glob_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    algs_lbl = algorithms_list.copy()
    for i in range(Numb_Algs):
        if(lamb[i] > 0):
            algorithms_list[i] = algorithms_list[i] + "_prox_" + str(lamb[i])
            algs_lbl[i] = algs_lbl[i] + "_prox"
        algorithms_list[i] = algorithms_list[i] + "_" + str(learning_rate[i]) + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b"
        train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(
            simple_read_data(loc_ep1[i], dataset + algorithms_list[i]))[:, :Numb_Glob_Iters]
        algs_lbl[i] = algs_lbl[i]

    plt.figure(1)
    linestyles = ['-', '--']
    algs_lbl = ["FEDL",  "FedAvg",
                "FEDL",  "FedAvg"]
    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    #min = train_loss.min()
    min = train_loss.min() - 0.01
    max = 3.5#train_loss.max() + 0.01
    num_al = 2
# Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    for i in range(num_al):
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$K_l = $' + str(loc_ep1[i]))
        ax1.set_ylim([min, max])
        ax1.legend(loc='upper right')

    for i in range(num_al):
        ax2.plot(train_loss[i+num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : " + '$K_l = $' + str(loc_ep1[i+num_al]))
        ax2.set_ylim([min, max])
        ax2.legend(loc='upper right')
    ax.set_title('FENIST', y=1.02)
    ax.set_xlabel('Global rounds ' + '$K_g$')
    ax.set_ylabel('Training Loss', labelpad=15)
    plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.pdf')
    plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.png')

    plt.figure(2)
    fig = plt.figure(figsize=(10, 4.5))
    #fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    max = glob_acc.max() + 0.01
    min = 0.1
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    
    for i in range(num_al):
        ax1.plot(glob_acc[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$K_l = $' + str(loc_ep1[i]))
        ax1.set_ylim([min, max])
        ax1.legend(loc='lower right')

    for i in range(num_al):
        ax2.plot(glob_acc[i+num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : " + '$K_l = $' + str(loc_ep1[i+num_al]))
        ax2.set_ylim([min, max])
        ax2.legend(loc='lower right')
    ax.set_title('FENIST', y=1.02)
    ax.set_xlabel('Global rounds ' + '$K_g$')
    ax.set_ylabel('Test Accuracy', labelpad=15)
    plt.savefig(dataset + str(loc_ep1[1]) + 'glob_acc.pdf')
    plt.savefig(dataset + str(loc_ep1[1]) + 'glob_acc.png')


def plot_summary_three_figures(num_users=100, loc_ep1=[], Numb_Glob_Iters=10, lamb=[], learning_rate=[], algorithms_list=[], batch_size=0, dataset=""):
    Numb_Algs = len(algorithms_list)
    train_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))
    glob_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    algs_lbl = algorithms_list.copy()
    for i in range(Numb_Algs):
        if(lamb[i] > 0):
            algorithms_list[i] = algorithms_list[i] + "_prox_" + str(lamb[i])
            algs_lbl[i] = algs_lbl[i] + "_prox"
        algorithms_list[i] = algorithms_list[i] + "_" + str(
            learning_rate[i]) + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b"
        train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(
            simple_read_data(loc_ep1[i], dataset + algorithms_list[i]))[:, :Numb_Glob_Iters]
        algs_lbl[i] = algs_lbl[i]

    plt.figure(1)

    linestyles = ['-', '--']
    algs_lbl = ["FEDL",  "FedAvg",
                "FEDL",  "FedAvg",
                "FEDL",  "FedAvg"]

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    #min = train_loss.min()
    min = train_loss.min() - 0.01
    max = 3.5  # train_loss.max() + 0.01
    num_al = 2
# Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    for i in range(num_al):
        stringbatch = str(batch_size[i])
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$K_l = $' + str(loc_ep1[i]) + ', $B = $' + stringbatch)
        ax1.set_ylim([min, max])
        ax1.legend(loc='upper right')


    for i in range(num_al):
        stringbatch = str(batch_size[i])
        ax2.plot(train_loss[i+num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : " + '$K_l = $' + str(loc_ep1[i+num_al]) + ', $B = $' + stringbatch)
        ax2.set_ylim([min, max])
        ax2.legend(loc='upper right')
    
    for i in range(num_al):
        stringbatch = str(batch_size[i])
        ax3.plot(train_loss[i+num_al*2, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al*2] + " : " + '$K_l = $' + str(loc_ep1[i+num_al*2]) + ', $B = $' + stringbatch)
        ax3.set_ylim([min, max])
        ax3.legend(loc='upper right')

    ax.set_title('FENIST', y=1.02)
    ax.set_xlabel('Global rounds ' + '$K_g$')
    ax.set_ylabel('Training Loss')
    plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.pdf', bbox_inches='tight')
    plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.png', bbox_inches='tight')

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    #min = train_loss.min()
    min = glob_acc.min() + 0.2
    max = glob_acc.max() + 0.01  # train_loss.max() + 0.01
    num_al = 2
# Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    for i in range(num_al):
        ax1.plot(glob_acc[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$K_l = $' + str(loc_ep1[i]) + ', $B = $' + stringbatch)
        ax1.set_ylim([min, max])
        ax1.legend(loc='lower right')

    for i in range(num_al):
        ax2.plot(glob_acc[i+num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : " + '$K_l = $' + str(loc_ep1[i+num_al]) + ', $B = $' + stringbatch)
        ax2.set_ylim([min, max])
        ax2.legend(loc='lower right')

    for i in range(num_al):
        ax3.plot(glob_acc[i+num_al*2, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al*2] + " : " + '$K_l = $' + str(loc_ep1[i+num_al*2]) + ', $B = $' + stringbatch)
        ax3.set_ylim([min, max])
        ax3.legend(loc='lower right')

    ax.set_title('FENIST', y=1.02)
    ax.set_xlabel('Global rounds ' + '$K_g$')
    ax.set_ylabel('Testing Accuracy', labelpad=15)
    plt.savefig(dataset + str(loc_ep1[1]) +
                'test_accu.pdf', bbox_inches='tight')
    plt.savefig(dataset + str(loc_ep1[1]) +
                'test_accu.png', bbox_inches='tight')


def plot_summary_three_figures_batch(num_users=100, loc_ep1=[], Numb_Glob_Iters=10, lamb=[], learning_rate=[], algorithms_list=[], batch_size=0, dataset=""):
    Numb_Algs = len(algorithms_list)
    train_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))
    glob_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    algs_lbl = algorithms_list.copy()
    for i in range(Numb_Algs):
        if(lamb[i] > 0):
            algorithms_list[i] = algorithms_list[i] + "_prox_" + str(lamb[i])
            algs_lbl[i] = algs_lbl[i] + "_prox"
        algorithms_list[i] = algorithms_list[i] + "_" + str(
            learning_rate[i]) + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b"
        train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(
            simple_read_data(loc_ep1[i], dataset + algorithms_list[i]))[:, :Numb_Glob_Iters]
        algs_lbl[i] = algs_lbl[i]

    plt.figure(1)

    linestyles = ['-', '--']
    algs_lbl = ["FEDL",  "FedAvg",
                "FEDL",  "FedAvg",
                "FEDL",  "FedAvg"]

    print("training loss")
    for i in range(6):
        print(train_loss[i].min())
    
    print("global accurancy")
    for i in range(6):
        print(glob_acc[i].max())
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    #min = train_loss.min()
    min = train_loss.min() - 0.01
    max = 0.7  # train_loss.max() + 0.01
    num_al = 2
# Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$K_l = $' + str(loc_ep1[i]) + ', $B = $' + stringbatch)
        ax1.set_ylim([min, max])
        ax1.legend(loc='upper right')

    for i in range(num_al):
        stringbatch = str(batch_size[i+num_al])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax2.plot(train_loss[i+num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : " + '$K_l = $' + str(loc_ep1[i+num_al]) + ', $B = $' + stringbatch)
        ax2.set_ylim([min, max])
        ax2.legend(loc='upper right')

    for i in range(num_al):
        stringbatch = str(batch_size[i+num_al*2])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax3.plot(train_loss[i+num_al*2, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al*2] + " : " + '$K_l = $' + str(loc_ep1[i+num_al*2]) + ', $B = $' + stringbatch)
        ax3.set_ylim([min, max])
        ax3.legend(loc='lower right')

    ax.set_title('MNIST', y=1.02)
    ax.set_xlabel('Global rounds ' + '$K_g$')
    ax.set_ylabel('Training Loss')
    plt.savefig(dataset + str(loc_ep1[1]) +
                'train_loss.pdf', bbox_inches='tight')
    plt.savefig(dataset + str(loc_ep1[1]) +
                'train_loss.png', bbox_inches='tight')
    
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    min = 0.8
    max = glob_acc.max() + 0.01 # train_loss.max() + 0.01
    num_al = 2
# Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax1.plot(glob_acc[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$K_l = $' + str(loc_ep1[i]) + ', $B = $' + stringbatch)
        ax1.set_ylim([min, max])
        ax1.legend(loc='lower right')

    for i in range(num_al):
        stringbatch = str(batch_size[i+num_al])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax2.plot(glob_acc[i+num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : " + '$K_l = $' + str(loc_ep1[i+num_al]) + ', $B = $' + stringbatch)
        ax2.set_ylim([min, max])
        ax2.legend(loc='lower right')

    for i in range(num_al):
        stringbatch = str(batch_size[i+num_al*2])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax3.plot(glob_acc[i+num_al*2, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al*2] + " : " + '$K_l = $' + str(loc_ep1[i+num_al*2]) + ', $B = $' + stringbatch)
        ax3.set_ylim([min, max])
        ax3.legend(loc='upper right')

    ax.set_title('MNIST', y=1.02)
    ax.set_xlabel('Global rounds ' + '$K_g$')
    ax.set_ylabel('Testing Accuracy', labelpad=15)
    plt.savefig(dataset + str(loc_ep1[1]) +
                'testing_accuracy.pdf', bbox_inches='tight')
    plt.savefig(dataset + str(loc_ep1[1]) +
                'testing_accuracy.png', bbox_inches='tight')

