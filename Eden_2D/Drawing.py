from matplotlib import pyplot as plt

def draw_square(x, y, col='gray', ls=0.35):
    """With center at x, y draw a square of area 1"""
    """it's area is actually 4ls^2, isn't it?"""
    """ls is a half of square side"""
    plt.fill([x - ls, x + ls, x + ls, x - ls], [y - ls, y - ls, y + ls, y + ls], color=col)


def draw_polyomino(eden, time):
    """This function draws a square for each (x, y) if the respective IO entry is 1 and does nothing if it is 0"""
    """As I got entries of eden are (x, y): [a, b, c] and if a == 1 than we draw the square with the center (x, y)
    b probably corresponds to the number how many times we could've add this square to the eden"""
    plt.style.use('ggplot')
    plt.axis('off')
    plt.rc('grid', linestyle="-", color='black')
    plt.grid(True)
    # plt.rc('grid', linestyle="-", color='black')  # why do we need this line twice? and does it change anything?
    plt.gca().set_aspect('equal', adjustable='box')

    for x in eden:
        if eden[x][0] == 1:
            draw_square(x[0], x[1], 'gray')
    draw_square(0, 0, 'green')
    plt.savefig('pictures/eden_' + str(time) + '.svg', format='svg', dpi=1200)
    plt.show()


def draw_polyomino_holes(eden, holes, time):
    """This function draws a square for each (x, y) if the respective IO entry is 1 and does nothing if it is 0"""
    plt.style.use('ggplot')
    plt.axis('off')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    for x in eden:
        if eden[x][0] == 1:
            draw_square(x[0], x[1], 'gray')
    for x in holes:
        s = len(holes[x])
        for i in range(0, s):
            draw_square(holes[x][i][0], holes[x][i][1], 'red')
    draw_square(0, 0, 'green')
    plt.savefig('pictures/eden_' + str(time) + '_holes.svg', format='svg', dpi=1200)
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
        if barcode[x][1][1] == 0:
            plt.plot([barcode[x][1][0], time], [i, i], 'k-', lw=2)
        else:
            plt.plot([barcode[x][1][0], barcode[x][1][1]], [i, i], 'k-', lw=2)
        i = i + 40
    fig.savefig('pictures/barcode_'+str(time)+'.svg', format='svg', dpi=1200)
    plt.show()
