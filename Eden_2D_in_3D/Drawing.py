from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import collections


def draw_square(x0, y0, z0, d, ax, alpha=0.5, col='gray', ls=0.45):
    """With center at x, y, z draw a square of area ls^2"""
    """d = 1 if square is parallel to xOy, d = 2 if x0z, d = 3 if y0z"""
    """ls is a half of square side"""
    if d == 0:
        # col = 'blue'
        y = np.linspace(y0-ls, y0+ls, num=2)
        z = y + z0 - y0
        y, z = np.meshgrid(y, z)
        x = np.ones((y.shape[0], y.shape[1])) * x0
        ax.plot_surface(x, y, z, color=col, alpha=alpha, linewidth=0, antialiased=True)
    if d == 1:
        # col = 'red'
        x = np.linspace(x0-ls, x0+ls, num=2)
        z = x + z0 - x0
        x, z = np.meshgrid(x, z)
        y = np.ones((x.shape[0], x.shape[1])) * y0
        ax.plot_surface(x, y, z, color=col, alpha=alpha, linewidth=0, antialiased=True)
    if d == 2:
        x = np.linspace(x0-ls, x0+ls, num=2)
        y = x + y0 - x0
        x, y = np.meshgrid(x, y)
        z = np.ones((x.shape[0], x.shape[1])) * z0
        ax.plot_surface(x, y, z, color=col, alpha=alpha, linewidth=0, antialiased=True)


def add_box(eden, ax, max_range=5):
    # Create cubic bounding box to simulate equal aspect ratio
    points = np.array([x for x in eden if eden[x][0] == 1])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    max_range = np.array([x.max()-x.min()+1, y.max()-y.min()+1, z.max()-z.min()+1]).max()/2

    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def draw_eden(eden, time):
    plt.style.use('ggplot')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.axis('off')
    # ax.grid(True)
    add_box(eden, ax)

    for x in eden:
        if eden[x][0] == 1:
            draw_square(x[0], x[1], x[2], x[3], ax=ax, col='gray')
    draw_square(0, 0, 0.5, 2, ax=ax, col='green')
    plt.savefig('pictures/eden_' + str(time) + '.svg', format='svg', dpi=1200)
    plt.show()


def draw_complex(eden, time, tile):
    plt.style.use('ggplot')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.axis('off')
    # ax.grid(True)

    add_box(eden, ax, 5)

    for x in eden:
        # if eden[x][0] == 1:
        draw_square(x[0], x[1], x[2], x[3], ax=ax, col='gray')
    # draw_square(0, 0, 0, 2, ax=ax, col='green')
    # draw_square(tile[0], tile[1], tile[2], tile[3], ax=ax, alpha=1, col='green')
    plt.savefig('pictures/eden_' + str(time) + '.svg', format='svg', dpi=1200)
    plt.show()


def draw_barcode(barcode, time):
    fig = plt.figure()
    plt.style.use('ggplot')
    # plt.axis('off')
    plt.grid(True)
    plt.rc('grid', linestyle="-", color='gray')
    plt.yticks([])
    plt.gca().set_aspect('equal', adjustable='box')
    i = 0
    for x in barcode:
        if x[1] == 0:
            plt.plot([x[0], time], [i, i], 'k-', lw=2)
        else:
            plt.plot([x[0], x[1]], [i, i], 'k-', lw=2)
        i = i + 40
    fig.savefig('pictures/barcode_'+str(time)+'.svg', format='svg', dpi=1200)
    plt.show()


def draw_frequencies_1(dict):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    l = len(dict[0])
    shift = 300
    ax.plot(range(shift, l), dict[-1][shift:], color='tab:red', label='-1')
    ax.plot(range(shift, l), dict[0][shift:], color='tab:orange', label='0')
    ax.plot(range(shift, l), dict[1][shift:], color='tab:green', label='1')
    ax.plot(range(shift, l), dict[2][shift:], color='tab:blue', label='2')

    plt.yscale('log')
    ax.set_title('betti_1 frequencies')
    ax.set_ylabel('frequency of change in betti_1')
    ax.set_xlabel('time')
    ax.legend()
    plt.show()
    fig.savefig('pictures/fr_b_1.png', format='png', dpi=1200)


def draw_frequencies_2(dict):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    l = len(dict[0])
    shift = 300
    ax.plot(range(shift, l), dict[0][shift:], color='tab:orange', label='0')
    ax.plot(range(shift, l), dict[1][shift:], color='tab:green', label='1')

    plt.yscale('log')
    ax.set_title('betti_2 frequencies')
    ax.set_ylabel('frequency of change in betti_2')
    ax.set_xlabel('time')
    ax.legend()
    plt.show()
    fig.savefig('pictures/fr_b_2.png', format='png', dpi=1200)


def draw_diagram_holes(created_holes, holes):
    fr_cr = [created_holes[i][-2] for i in range(len(created_holes))]
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
    ax.bar(x - width/2, counter_cr.values(), width, label='Total')
    ax.bar(x + width/2, counter_final.values(), width, label='Final')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Frequency of Number of Holes')
    ax.set_xlabel('Volume')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.show()
    fig.savefig('pictures/holes.png', format='png', dpi=1200)


def draw_tri_tetra(tri, tri_f, tetra, tetra_f):
    width = 0.6
    labels = list(tri)+list(tetra)
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    plt.yscale('log')

    ax.bar(x[:2], tri.values(), width, label='Tricubes Total', color='navy')
    ax.bar(x[:2], tri_f.values(), width, label='Tricubes Final', color='royalblue')
    ax.bar(x[2:], tetra.values(), width, label='Tetracubes Total', color='chocolate')
    ax.bar(x[2:], tetra_f.values(), width, label='Tetracubes Final', color='orange')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Frequency of Number of Holes')
    ax.set_xlabel('Volume')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()

    plt.show()
    fig.savefig('pictures/tri-tetra-cubes.png', format='png', dpi=1200)




