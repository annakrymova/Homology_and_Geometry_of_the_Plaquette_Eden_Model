from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import collections
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches


def draw_diagram_holes(created_holes, holes, time):
    fr_cr = [created_holes[i][-1] for i in range(len(created_holes))]
    fr_cr.sort()
    fr_final = [len(holes[i]) for i in holes]
    fr_final.sort()
    counter_cr = collections.Counter(fr_cr)
    counter_final = collections.Counter(fr_final)
    labels = []
    a = np.arange(1, len(counter_cr) + 1)
    for i in a:
        labels.append(str(i))

    for i in counter_cr.keys():
        if i not in counter_final.keys():
            counter_final[i] = 0

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    plt.yscale('log')
    ax.bar(x - width/2, counter_cr.values(), width, color=[(0.44, 0.57, 0.79)], label='Total')
    ax.bar(x + width/2, counter_final.values(), width, color=[(225/256, 151/256, 76/256)], label='Final')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Frequency of Number of Holes')
    ax.set_xlabel('Volume')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if len(labels) >= 50:
        plt.setp(ax.get_xticklabels(), fontsize=4)
    elif len(labels) >= 45:
        plt.setp(ax.get_xticklabels(), fontsize=5)
    elif len(labels) >= 40:
        plt.setp(ax.get_xticklabels(), fontsize=6)
    elif len(labels) >= 30:
        plt.setp(ax.get_xticklabels(), fontsize=7)
    elif len(labels) >= 20:
        plt.setp(ax.get_xticklabels(), fontsize=6)

    ax.legend()

    fig.tight_layout()

    plt.show()
    fig.savefig('pictures/'+str(int(time/1000))+'k/holes.png', format='png', dpi=1200)


def plot_b_per(b1, p2, time, N=0):
    n = 0
    nn = 0

    def func(x, a, b):
        return a * x ** b
    ydata_f = b1
    xdata_f = range(len(ydata_f))
    ydata = ydata_f[N:]
    xdata = xdata_f[N:]
    # plt.xscale('log')
    # plt.yscale('log')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.plot(xdata_f[n:], ydata_f[n:], 'm-', label=r'$\beta_1(t)$ data',  linewidth=0.75)
    # try:
    popt, pcov = curve_fit(func, xdata, ydata)
    # except:
    #     popt, pcov = curve_fit(func, xdata, ydata, bounds=([0., 0., -10], [10., 2., 900]))

    plt.plot(xdata_f[n:], func(xdata_f[n:], *popt), 'm--', label=r'fit: $y=%5.2f x^{%5.3f}$' % tuple(popt), linewidth=0.75)

    ydata = p2
    xdata = range(len(ydata))
    plt.plot(xdata[n:], ydata[n:], color='orange', linestyle='solid', label=r'$P_{2}(t)$ data',  linewidth=0.75)
    popt, pcov = curve_fit(func, xdata, ydata)
    plt.plot(xdata[n:], func(xdata[n:], *popt), color='orange', linestyle='dashed', label=r'fit: $y=%5.2f x^{%5.3f}$' % tuple(popt),  linewidth=0.75)

    plt.xlabel('t')
    plt.ylabel('data')
    plt.legend(prop={'size': 6})
    plt.tight_layout()
    plt.savefig('pictures/'+str(int(time/1000))+'k/per-b-time_'+str(time)+'.png', dpi=1200)
    plt.show()


def draw_tri_tetra(tri, tri_f, tetra, tetra_f, time):
    width = 0.35
    labels = list(tri)+list(tetra)
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    plt.yscale('log')

    ax.bar(x[:2]-width/2, tri.values(), width, label='Trominoes Total', color='navy')

    ax.bar(x[2:]-width/2, tetra.values(), width, label='Tetrominoes Total', color='chocolate')
    ax.bar(x[:2]+width/2, tri_f.values(), width, label='Trominoes Final', color='royalblue')

    try:
        ax.bar(x[2:]+width/2, tetra_f.final(), width, label='Tetrominoes Final', color='orange')
        ax.legend(loc='upper right')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    except:
        # ax.bar(x[2:]+width/2, [0,0,0,0,0.00001], width, label='Tetrominoes Final', color='orange')
        print('oops')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        handles, labels = ax.get_legend_handles_labels()
        labels = [labels[0], labels[2], labels[1]]
        patch = mpatches.Patch(color='orange', label='Tetrominoes Final', linewidth=0.35)
        handles.append(patch)
        labels.append('Tetrominoes Final')

        ax.legend(handles, labels, loc='upper right')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Frequency of Number of Holes')
    ax.set_xlabel('Type of a Hole')


    fig.tight_layout()

    plt.show()
    fig.savefig('pictures/'+str(int(time/1000))+'k/tro-tetro-minoes.png', format='png', dpi=1200)

