from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def draw_square(x0, y0, z0, d, ax, col='gray', ls=0.45):
    """With center at x, y, z draw a square of area ls^2"""
    """d = 1 if square is parallel to xOy, d = 2 if x0z, d = 3 if y0z"""
    """ls is a half of square side"""
    if d == 0:
        x = np.arange(x0-ls, x0+ls, 0.05)
        y = x + y0 - x0
        x, y = np.meshgrid(x, y)
        z = np.ones((x.shape[0], x.shape[0])) * z0
        ax.plot_surface(x, y, z, color=col, linewidth=0, antialiased=False)
    if d == 1:
        x = np.arange(x0-ls, x0+ls, 0.05)
        z = x + z0 - x0
        x, z = np.meshgrid(x, z)
        y = np.ones((x.shape[0], x.shape[0])) * y0
        ax.plot_surface(x, y, z, color=col, linewidth=0, antialiased=False)
    if d == 2:
        y = np.arange(y0-ls, y0+ls, 0.05)
        z = y + z0 - y0
        y, z = np.meshgrid(y, z)
        x = np.ones((y.shape[0], y.shape[0])) * x0
        ax.plot_surface(x, y, z, color=col, linewidth=0, antialiased=False)


def add_box(max_range):
    # Create cubic bounding box to simulate equal aspect ratio
    x = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten()
    y = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten()
    z = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten()
    # Comment or uncomment following both lines to test the fake bounding box:
    for x, y, z in zip(x, y, z):
        plt.plot([x], [y], [z], 'w')


def draw(eden, time):
    plt.style.use('ggplot')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.axis('off')
    ax.grid(True)

    max_range = np.absolute(np.array(list(eden.keys()))[:, :3]).max() + 1
    add_box(max_range)

    for x in eden:
        if eden[x][0] == 1 or eden[x][0] == 0:
            draw_square(x[0], x[1], x[2], x[3], ax=ax, col='gray')
    draw_square(0, 0, 0, 0, ax=ax, col='green')
    plt.savefig('pictures/eden_' + str(time) + '.svg', format='svg', dpi=1200)
    plt.show()

