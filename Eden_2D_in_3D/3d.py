#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 19:34:33 2019

@author: err
"""
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import gudhi as gd
import numpy as np

try:
    from line_profiler import LineProfiler

    def do_profile(follow=[]):
        def inner(func):
            def profiled_func(*args, **kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)
                    for f in follow:
                        profiler.add_function(f)
                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats()
            return profiled_func
        return inner

except ImportError:
    def do_profile(follow=[]):
        "Helpful if you accidentally leave in production!"
        def inner(func):
            def nothing(*args, **kwargs):
                return func(*args, **kwargs)
            return nothing
        return inner
################


def hamming2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def Diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif

def draw_barcode(barcode, time):
    """ """
    fig = plt.figure()
    plt.style.use('ggplot')
    # plt.axis('off')
    # plt.rc('grid', linestyle="-", color='black')
    plt.grid(True)
    plt.rc('grid', linestyle="-", color='gray')
    plt.yticks([])
    plt.gca().set_aspect('equal', adjustable='box')
    i = 0
    for x in barcode:
        if barcode[x][1] == 0:
            plt.plot([barcode[x][0], time], [i, i], 'k-', lw=2)
        else:
            plt.plot([barcode[x][0], barcode[x][1]], [i, i], 'k-', lw=2)
        i = i + 40
    fig.savefig('5000.png')
    plt.show()

def start_eden():
    eden = {(0, 0, 0): [1, 0, 0],
            (0, 0, 1): [0, 1, 0],
            (0, 0, -1): [0, 1, 0],
            (1, 0, 0): [0, 1, 0],
            (-1, 0, 0): [0, 1, 0],
            (0, 1, 0): [0, 1, 0],
            (0, -1, 0): [0, 1, 0]}
    perimeter = [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
    return eden, perimeter

def actualize_neighbors(tile_selected, eden, perimeter):
    n3 = [tile_selected[0] + 1, tile_selected[1], tile_selected[2]]
    n4 = [tile_selected[0] + -1, tile_selected[1], tile_selected[2]]
    n5 = [tile_selected[0], tile_selected[1] + 1, tile_selected[2]]
    n6 = [tile_selected[0], tile_selected[1] - 1, tile_selected[2]]
    n1 = [tile_selected[0], tile_selected[1], tile_selected[2] + 1]
    n2 = [tile_selected[0], tile_selected[1], tile_selected[2] - 1]

    n1 = tuple(n1)
    n2 = tuple(n2)
    n3 = tuple(n3)
    n4 = tuple(n4)
    n5 = tuple(n5)
    n6 = tuple(n6)

    nearest_n_tiles = [n1, n2, n3, n4, n5, n6]

    nearest_n = [0, 0, 0, 0, 0, 0]

    if n1 in eden:
        eden[n1][1] = eden[n1][1] + 1
        if eden[n1][0] == 1:
            nearest_n[0] = 1
    else:
        eden[n1] = [0, 1, eden[tile_selected][2]]
        perimeter = perimeter + [n1]

    if n2 in eden:
        eden[n2][1] = eden[n2][1] + 1
        if eden[n2][0] == 1:
            nearest_n[1] = 1
    else:
        eden[n2] = [0, 1, eden[tile_selected][2]]
        perimeter = perimeter + [n2]

    if n3 in eden:
        eden[n3][1] = eden[n3][1] + 1
        if eden[n3][0] == 1:
            nearest_n[2] = 1
    else:
        eden[n3] = [0, 1, eden[tile_selected][2]]
        perimeter = perimeter + [n3]

    if n4 in eden:
        eden[n4][1] = eden[n4][1] + 1
        if eden[n4][0] == 1:
            nearest_n[3] = 1
    else:
        eden[n4] = [0, 1, eden[tile_selected][2]]
        perimeter = perimeter + [n4]

    if n5 in eden:
        eden[n5][1] = eden[n5][1] + 1
        if eden[n5][0] == 1:
            nearest_n[4] = 1
    else:
        eden[n5] = [0, 1, eden[tile_selected][2]]
        perimeter = perimeter + [n5]

    if n6 in eden:
        eden[n6][1] = eden[n6][1] + 1
        if eden[n6][0] == 1:
            nearest_n[5] = 1
    else:
        eden[n6] = [0, 1, eden[tile_selected][2]]
        perimeter = perimeter + [n6]

    return eden, perimeter, nearest_n, nearest_n_tiles

def neighbours(eden, tile_selected):
    ###### For the Euler Characteristic

    #### Level one z = z -1, Level two z = 0, Level 3 Z = +1

    l11 = [tile_selected[0] - 1, tile_selected[1] + 1, tile_selected[2] - 1]
    l16 = [tile_selected[0], tile_selected[1] + 1, tile_selected[2] - 1]
    l12 = [tile_selected[0] + 1, tile_selected[1] + 1, tile_selected[2] - 1]
    l15 = [tile_selected[0] - 1, tile_selected[1], tile_selected[2] - 1]
    l17 = [tile_selected[0] + 1, tile_selected[1], tile_selected[2] - 1]
    l13 = [tile_selected[0] - 1, tile_selected[1] - 1, tile_selected[2] - 1]
    l18 = [tile_selected[0], tile_selected[1] - 1, tile_selected[2] - 1]
    l14 = [tile_selected[0] + 1, tile_selected[1] - 1, tile_selected[2] - 1]

    l21 = [tile_selected[0] - 1, tile_selected[1] + 1, tile_selected[2]]
    l22 = [tile_selected[0] + 1, tile_selected[1] + 1, tile_selected[2]]
    l23 = [tile_selected[0] - 1, tile_selected[1] - 1, tile_selected[2]]
    l24 = [tile_selected[0] + 1, tile_selected[1] - 1, tile_selected[2]]

    l31 = [tile_selected[0] - 1, tile_selected[1] + 1, tile_selected[2] + 1]
    l36 = [tile_selected[0], tile_selected[1] + 1, tile_selected[2] + 1]
    l32 = [tile_selected[0] + 1, tile_selected[1] + 1, tile_selected[2] + 1]
    l35 = [tile_selected[0] - 1, tile_selected[1], tile_selected[2] + 1]
    l37 = [tile_selected[0] + 1, tile_selected[1], tile_selected[2] + 1]
    l33 = [tile_selected[0] - 1, tile_selected[1] - 1, tile_selected[2] + 1]
    l38 = [tile_selected[0], tile_selected[1] - 1, tile_selected[2] + 1]
    l34 = [tile_selected[0] + 1, tile_selected[1] - 1, tile_selected[2] + 1]

    nn = [tuple(l11), tuple(l16), tuple(l12), tuple(l15), tuple(l17), tuple(l13), tuple(l18), tuple(l14),
          tuple(l21), tuple(l22), tuple(l23), tuple(l24), tuple(l31), tuple(l36), tuple(l32), tuple(l35),
          tuple(l37), tuple(l33), tuple(l38), tuple(l34)]
    n = [0] * 20
    for i in range(0, len(nn)):
        if nn[i] in eden:
            if eden[nn[i]][0] == 1:
                n[i] = 1
    return n

def actualize_vef(vertices, edges, faces, nearest_n, n):
    v = [1] * 8
    e = [1] * 12
    f = [1] * 6

    if n[0] == 1:
        v[0] = 0
    if n[1] == 1:
        v[0] = 0
        v[1] = 0
        e[0] = 0
    if n[2] == 1:
        v[1] = 0
    if n[3] == 1:
        v[0] = 0
        v[2] = 0
        e[3] = 0
    if n[4] == 1:
        v[1] = 0
        v[3] = 0
        e[1] = 0
    if n[5] == 1:
        v[2] = 0
    if n[6] == 1:
        v[2] = 0
        v[3] = 0
        e[2] = 0
    if n[7] == 1:
        v[3] = 0

    if n[8] == 1:
        v[0] = 0
        v[4] = 0
        e[11] = 0
    if n[9] == 1:
        v[1] = 0
        v[5] = 0
        e[8] = 0
    if n[10] == 1:
        v[2] = 0
        v[6] = 0
        e[10] = 0
    if n[11] == 1:
        v[3] = 0
        v[7] = 0
        e[9] = 0

    if n[12] == 1:
        v[4] = 0
    if n[13] == 1:
        v[4] = 0
        v[5] = 0
        e[4] = 0
    if n[14] == 1:
        v[5] = 0
    if n[15] == 1:
        v[4] = 0
        v[6] = 0
        e[7] = 0
    if n[16] == 1:
        v[5] = 0
        v[7] = 0
        e[5] = 0
    if n[17] == 1:
        v[6] = 0
    if n[18] == 1:
        v[6] = 0
        v[7] = 0
        e[6] = 0
    if n[19] == 1:
        v[7] = 0

    if nearest_n[0] == 1:
        v[4] = 0
        v[5] = 0
        v[7] = 0
        v[6] = 0
        e[4] = 0
        e[5] = 0
        e[6] = 0
        e[7] = 0
        f[0] = 0
    if nearest_n[1] == 1:
        v[0] = 0
        v[1] = 0
        v[2] = 0
        v[3] = 0
        e[0] = 0
        e[1] = 0
        e[2] = 0
        e[3] = 0
        f[1] = 0
    if nearest_n[2] == 1:
        v[1] = 0
        v[5] = 0
        v[3] = 0
        v[7] = 0
        e[1] = 0
        e[5] = 0
        e[8] = 0
        e[9] = 0
        f[2] = 0
    if nearest_n[3] == 1:
        v[0] = 0
        v[4] = 0
        v[2] = 0
        v[6] = 0
        e[3] = 0
        e[7] = 0
        e[11] = 0
        e[10] = 0
        f[3] = 0
    if nearest_n[4] == 1:
        v[0] = 0
        v[4] = 0
        v[1] = 0
        v[5] = 0
        e[0] = 0
        e[4] = 0
        e[11] = 0
        e[8] = 0
        f[4] = 0
    if nearest_n[5] == 1:
        v[2] = 0
        v[6] = 0
        v[3] = 0
        v[7] = 0
        e[2] = 0
        e[6] = 0
        e[9] = 0
        e[10] = 0
        f[5] = 0

    vertices = vertices + sum(v)
    edges = edges + sum(e)
    faces = faces + sum(f)

    return vertices, edges, faces, sum(v), sum(e), sum(f)

def euler_characteristic(k0, k1, k2, k3):
    return k0 - k1 + k2 - k3

def add_neighbours_bds(bds, j, iterations, num_possible_components, merged, finished, eden):
    tile_selected = bds[j][iterations]

    n3 = [tile_selected[0] + 1, tile_selected[1], tile_selected[2]]
    n4 = [tile_selected[0] + -1, tile_selected[1], tile_selected[2]]
    n5 = [tile_selected[0], tile_selected[1] + 1, tile_selected[2]]
    n6 = [tile_selected[0], tile_selected[1] - 1, tile_selected[2]]
    n1 = [tile_selected[0], tile_selected[1], tile_selected[2] + 1]
    n2 = [tile_selected[0], tile_selected[1], tile_selected[2] - 1]

    n1 = tuple(n1)
    n2 = tuple(n2)
    n3 = tuple(n3)
    n4 = tuple(n4)
    n5 = tuple(n5)
    n6 = tuple(n6)

    nearest_n_tiles = [n1, n2, n3, n4, n5, n6]

    nearest_n = [0, 0, 0, 0, 0, 0]

    if n1 in eden:
        if eden[n1][0] == 0:
            nearest_n[0] = 1
    else:
        nearest_n[0] = 1

    if n2 in eden:
        if eden[n2][0] == 0:
            nearest_n[1] = 1
    else:
        nearest_n[1] = 1

    if n3 in eden:
        if eden[n3][0] == 0:
            nearest_n[2] = 1
    else:
        nearest_n[2] = 1

    if n4 in eden:
        if eden[n4][0] == 0:
            nearest_n[3] = 1
    else:
        nearest_n[3] = 1

    if n5 in eden:
        if eden[n5][0] == 0:
            nearest_n[4] = 1
    else:
        nearest_n[4] = 1

    if n6 in eden:
        if eden[n6][0] == 0:
            nearest_n[5] = 1
    else:
        nearest_n[5] = 1

    for i in range(0, 6):
        if nearest_n[i] == 1:
            if nearest_n_tiles[i] not in bds[j]:
                bds[j] = bds[j] + [nearest_n_tiles[i]]
            for t in range(0, num_possible_components):
                if nearest_n_tiles[i] in bds[t]:
                    if t < j:
                        merged[j] = 1
                        finished[j] = 1
                    if t > j:
                        merged[t] = 1
                        finished[t] = 1
    return bds, merged, finished

def increment_betti_2(eden, tile_selected, nearest_n, nearest_n_tiles, barcode, time, holes, total_holes, created_holes,
                      tags, model):

    total_holes_old = total_holes
    barcode_old = barcode
    holes_old = holes
    created_holes_old = created_holes
    tags_old = tags

    if eden[tile_selected][2] == 0:
        per = 1  # This is 1 if the tile added was in the out perimeter
    else:
        num_hole = eden[tile_selected][2]
        per = 0
    # In this case the tile added was in a hole

    betti_2 = 0

    if sum(nearest_n) == 6:
        betti_2 = - 1
        barcode[num_hole][1][1] = float(time + 2)  # Are we covering a hole that was never divided?
        holes[num_hole].remove(tile_selected)
        if holes[num_hole] == []:
            holes.pop(num_hole)

    if sum(nearest_n) == 5:
        betti_2 = 0
        if per == 0:
            holes[num_hole].remove(tile_selected)
    # print(nearest_n)
    if sum(nearest_n) != 6 and sum(nearest_n) != 5:
        num_possible_components = 0
        bds = []
        iterations = 0
        for i in range(0, 6):
            if nearest_n[i] == 0:
                num_possible_components = num_possible_components + 1
                bds = bds + [[nearest_n_tiles[i]]]

        finished = [0] * num_possible_components
        merged = finished.copy()

        while sum(finished) < num_possible_components - per:
            for j in range(0, num_possible_components):
                if finished[j] == 0:
                    bds, merged, finished = add_neighbours_bds(bds, j, iterations, num_possible_components, merged,
                                                               finished, eden)
                    if (iterations + 1) == len(bds[j]):
                        finished[j] = 1
            iterations = iterations + 1

        betti_2 = (num_possible_components - 1) - sum(merged)

        if model == 'no_betti_2' and betti_2 >= 1:
            return betti_2, total_holes_old, eden, barcode_old, holes_old, created_holes_old, tags_old

        # print(betti_2, per)
        # At this point we have the bds components and the ones that were not merged will become the holes.
        # Here we actualize the holes and we actualize Hole No in eden.
        if betti_2 == 0:
            if per == 0:
                holes[num_hole].remove(tile_selected)

        else:
            # if model == 'no_betti_2':
                # return betti_2, total_holes, eden, barcode, holes, created_holes, tags
            if per == 1:
                for i in range(0, num_possible_components):
                    if finished[i] == 1 and merged[i] == 0:
                        total_holes = total_holes + 1
                        holes[total_holes] = bds[i].copy()

                        for x in bds[i]:
                            if x in eden:
                                eden[x][2] = total_holes

                        barcode[total_holes] = [0, [float(time + 2), float(0)], [total_holes]]
                        created_holes = created_holes + [[barcode[total_holes][2], bds[i].copy(), len(bds[i])]]

            else:
                if barcode[num_hole][0] == 0:
                    tags = tags + [num_hole]
                    barcode[num_hole][0] = 1

                holes.pop(num_hole)

                for i in range(0, num_possible_components):
                    if finished[i] == 1 and merged[i] == 0:
                        total_holes = total_holes + 1
                        holes[total_holes] = bds[i].copy()
                        for x in bds[i]:
                            if x in eden:
                                eden[x][2] = total_holes
                        barcode[total_holes] = [1, [float(time + 2), float(0)], barcode[num_hole][2] + [total_holes]]
                        created_holes = created_holes + [[barcode[total_holes][2], bds[i].copy(), len(bds[i])]]

    return betti_2, total_holes, eden, barcode, holes, created_holes, tags

def return_betti_1(betti_2, euler_ch):
    # X = V - E + F - Cubes
    # X = b_0 - b_1 + b_2
    # Observe that b_0 is 1 because polyominoes are connected
    # Cubes us exactly the time because we add one cube at each time
    return 1 + betti_2 - euler_ch

def barcode_forest(barcode, tags):
    bars_pure = []
    bars_hole = []
    for x in barcode:
        if barcode[x][0] == 0:
            bars_pure = bars_pure + [barcode[x][1]]

    for x in tags:
        b = {}
        for elem in barcode:
            if barcode[elem][2][0] == x:
                b[tuple(barcode[elem][2])] = barcode[elem][1]
        #        print(b)
        bars_hole = bars_hole + bars_from_tree(b, x)
    return bars_pure + bars_hole

def bars_from_tree(b, tag):
    n = max(len(x) for x in b)
    bars = []
    while n > 0:
        leaves_parent = [x for x in b if len(x) == n - 1]

        #        print(leaves_parent)
        possible_leaves = [x for x in b if len(x) == n]
        #        print(possible_leaves)
        for j in leaves_parent:
            leaves = []
            for x in possible_leaves:
                #            print(x)
                root = list(x)
                del root[-1]
                root = tuple(root)
                if hamming2(j, root) == 0:
                    leaves = leaves + [x]
            #            Diff(possible_leaves, leaves)
            #            print(leaves)
            if len(leaves) > 0:
                times = []
                for x in leaves:
                    times = times + [b[x][1]]
                if 0 in times:
                    ind = times.index(0)
                    for i in range(0, len(leaves)):
                        if i != ind:
                            bars = bars + [b[leaves[i]]]
                else:
                    ind = times.index(max(times))
                    for i in range(0, len(leaves)):
                        if i != ind:
                            bars = bars + [b[leaves[i]]]
                    b[j][1] = max(times)
        n = n - 1
    bars = bars + [b[(tag,)]]
    return bars

def convert_perseus(process):
    min_x = min(process, key=lambda x: x[0])
    max_x = max(process, key=lambda x: x[0])
    min_y = min(process, key=lambda x: x[1])
    max_y = max(process, key=lambda x: x[1])
    min_z = min(process, key=lambda x: x[2])
    max_z = max(process, key=lambda x: x[2])
    dimension = 3
    long = max_x[0] + ((-1) * (min_x[0])) + 1
    wide = max_y[1] + ((-1) * (min_y[1])) + 1
    deep = max_z[2] + ((-1) * (min_z[2])) + 1
    with open('300000_3D_1_final.txt', 'w') as f:
        f.writelines('%d\n' % dimension)
        f.writelines('%d\n' % long)
        f.writelines('%d\n' % wide)
        f.writelines('%d\n' % deep)
        for z in range(min_z[2], max_z[2] + 1):
            for i in range(min_y[1], max_y[1] + 1):
                for j in range(min_x[0], max_x[0] + 1):
                    #                   print((j,i, z))
                    if (j, i, z) in process:
                        f.writelines('%d\n' % process.index((j, i, z)))
                    #                       print(process.index((j,i,z))+1)
                    else:
                        f.writelines('inf\n')

def convert_perseus_2(process):
    dimension = 3
    with open('1000000_3D_12_final.txt', 'w') as f:
        f.writelines('%d\n' % dimension)
        i = 0
        for x in process:
            i = i + 1
            y = (x[0], x[1], x[2], i)
            f.writelines('%s %s %s %s\n' % y)


##########################

# @do_profile(follow=[increment_betti_2])
def grow_eden(t=10000, model='no_betti_2_new'):
    vertices = 8
    edges = 12
    faces = 6

    process = [(0, 0, 0)]
    perimeter_len = [6]
    eden, perimeter = start_eden()

    l = len(perimeter)

    holes = {}
    total_holes = 0
    barcode = {}
    tags = []
    created_holes = []

    betti_2_total = 0
    betti_2_vector_changes = [0]
    betti_2_total_vector = [0]

    betti_1_total_vector = [0]

    skipped = 0
    size = 1
    euler_char_prev = 1

    pbar = tqdm(total=t, position=0, leave=True)
    pbar.update(1)
    # for i in tqdm(range(1, t)):
    # for i in (range(1, t)):
    while size < t:
        # eden_old = copy.deepcopy(eden)
        # perimeter_old = copy.deepcopy(perimeter)
        # print(size)
        l = len(perimeter)

        x = random.randint(0, l - 1)
        tile_selected = perimeter[x]
        perimeter.pop(x)

        eden[tile_selected][0] = 1
        process = process + [tile_selected]

        eden, perimeter, nearest_n, nearest_n_tiles = actualize_neighbors(tile_selected, eden, perimeter)
        n = neighbours(eden, tile_selected)

        vertices, edges, faces, v_new, e_new, f_new = actualize_vef(vertices, edges, faces, nearest_n, n)
        euler_character = euler_characteristic(vertices, edges, faces, size + 1)

        if (model == 'no_betti_2_new' and euler_character <= euler_char_prev) or (model != 'no_betti_2_new'):
          betti_2, total_holes, eden, barcode, holes, created_holes, tags = increment_betti_2(eden, tile_selected,
                                                                                            nearest_n, nearest_n_tiles,
                                                                                            barcode, size, holes,
                                                                                            total_holes, created_holes,
                                                                                            tags, model)

        if (euler_character > euler_char_prev and model == 'no_betti_2_new') or (betti_2 >= 1 and model == 'no_betti_2'):
            skipped += 1
            perimeter += [tile_selected]
            for i, tile in enumerate(nearest_n_tiles):
                eden[tile][1] -= 1
            eden[tile_selected][0] = 0
            # print(eden[tile_selected][0])
            del process[-1]
            vertices = vertices - v_new
            edges = edges - e_new
            faces = faces - f_new
            euler_character = euler_char_prev
            continue

        pbar.update(1)

        size += 1
        betti_2_vector_changes += [betti_2]
        betti_2_total += betti_2
        betti_2_total_vector += [betti_2_total]

        betti_1_total = return_betti_1(betti_2_total, euler_character)
        betti_1_total_vector += [betti_1_total]

        l = len(perimeter)
        perimeter_len = perimeter_len + [l]
        euler_char_prev = euler_character

    final_barcode = barcode_forest(barcode, tags)

    pbar.close()
    return eden, perimeter, betti_2_total_vector, betti_2_vector_changes, barcode, holes,\
           betti_1_total, betti_1_total_vector, created_holes, \
           process, perimeter_len, skipped, size, final_barcode

# result = grow_eden()


Time = 10000
Model = 'no_betti_2_new'
Eden, Perimeter, Betti_2_total_vector, Betti_2_vector_changes, Barcode, Holes, Betti_1_total, \
    Betti_1_total_vector, Created_holes, Process, Perimeter_len, Skipped, I, Final_barcode = grow_eden(Time, Model)

convert_perseus(Process)

eden_model = gd.CubicalComplex(perseus_file='300000_3D_1_final.txt')

eden_model.persistence()
A = eden_model.persistence_intervals_in_dimension(1)
B = [elem for elem in A if elem[1] == float('inf')]
A = eden_model.persistence_intervals_in_dimension(2)
B = [elem for elem in A if elem[1] == float('inf')]
final = np.array(Final_barcode)
A_sorted = A.sort()
final_sorted = final.sort()
print(A_sorted == final_sorted)
# gd.plot_persistence_diagram(eden_model.persistence(), legend=True)
# gd.plot_persistence_barcode(eden_model.persistence())
gd.plot_persistence_barcode(eden_model.persistence(), legend=True)
a = 10
