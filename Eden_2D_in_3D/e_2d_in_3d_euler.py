import matplotlib.pyplot as plt
import numpy as np

def draw_frequencies_2(dict, changes, folder_name):
    fig, ax = plt.subplots()
    l = len(dict[0])

    ch_1 = [i for i, j in enumerate(changes) if j == 1]
    y_1 = []
    for x in ch_1:
        y_1 += [dict[1][x+1]]

    if next((i for i, x in enumerate(dict[1]) if x), 0) != 0:
        ax.scatter(ch_1, y_1, s=5, marker='o', color="tab:purple", label='+1')

    plt.yscale('log')
    ax.set_ylabel(r'Frequency of Change in $\beta_2$')
    ax.set_xlabel('t')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.legend(loc=1, prop={'size': 6})
    fig.savefig(folder_name+'/fr_b_2.png', format='png', dpi=1200)
    plt.close()

def draw_frequencies_1(dict, changes, folder_name):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    l = len(dict[0])

    ch_1 = [i for i, j in enumerate(changes) if j == -1]
    y_1 = []
    for x in ch_1:
        y_1 += [dict[1][x+1]]

    sh = []
    for j in np.arange(0, 3):
        sh.append(next((i for i, x in enumerate(dict[j]) if x), 0))
    shift = max(sh)

    # shiftt = next((i for i, x in enumerate(dict[-3]) if x), 0)
    # ax.plot(range(shiftt, l), dict[-3][shiftt:], color='tab:olive', label='-3',  linewidth=0.75)
    # ax.plot(range(shift, l), dict[-2][shift:], color='black', label='-2',  linewidth=0.75)
    # ax.plot(range(shift, l), dict[-1][shift:], color='tab:red', label='-1',  linewidth=0.75)
    ax.plot(range(shift, l), dict[0][shift:], color='tab:orange', label='0',  linewidth=0.75)
    ax.plot(range(shift, l), dict[1][shift:], color='tab:green', label='+1',  linewidth=0.75)
    ax.plot(range(shift, l), dict[2][shift:], color='tab:blue', label='+2',  linewidth=0.75)
    # shift = next((i for i, x in enumerate(dict[3]) if x), 0)
    # ax.plot(range(shift, l), dict[3][shift:], color='tab:purple', label='+3',  linewidth=0.75)
    if next((i for i, x in enumerate(dict[-1]) if x), 0) != 0:
        plt.scatter(ch_1, y_1, s=5, marker='o', color="tab:red", label='-1')

    plt.yscale('log')
    ax.set_ylabel(r'Frequency of Change in $\beta_1$')
    ax.set_xlabel('t')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.legend(loc=1, prop={'size': 6})
    fig.savefig(folder_name+'/fr_b_1.png', format='png', dpi=1200)
    plt.close()
