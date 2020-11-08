from tqdm import tqdm
import random
import numpy as np
from Supplementary_functions import start_eden_2d_in_3d, actualize_neighbors, neighbours_diag, actualize_vef, \
    nearest_cubes, nearest_voids, dimension, shift_for_neighbors, shift_for_neighbours_diag, \
    check_cube_in_eden, update_void_dict, shift_vertices, return_vertices, euler_characteristic, \
    return_betti_1, edge_voids, shift_for_edge_voids, barcode_forest, num_holes, tiles_from_voids, \
    final_inner_2d, start_eden_2d_in_3d_line

from Drawing import draw_eden, draw_complex, draw_square, draw_barcode, draw_frequencies_1, draw_frequencies_2, \
    draw_diagram_holes, draw_tri_tetra, plot_b_per
import itertools
import sys
from itertools import groupby
import collections
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection


def grow_eden(t, model, length):
    vertices = 4
    edges = 4
    # if model == 'line':
    #     eden, perimeter = start_eden_2d_in_3d_line(length)
    # else:
    #
    eden, perimeter = start_eden_2d_in_3d()
    # perimeter is an array consisting of all tiles that are on the perimeter
    inner_perimeter = []
    shift_for_vertices = shift_vertices(0), shift_vertices(1), shift_vertices(2)
    # process = [return_vertices((0, 0, 0, 2), shift_for_vertices)]  # an array consisting of all tiles that were added (specified by there 4 vertices)
    process = 0
    perimeter_len = []  # an array consisting of perimeter lengths at every time step

    shift_neighbours = [shift_for_neighbors(0), shift_for_neighbors(1), shift_for_neighbors(2)]
    shift_diag_neighbours = [shift_for_neighbours_diag(0), shift_for_neighbours_diag(1), shift_for_neighbours_diag(2)]
    shift_edge_voids = [shift_for_edge_voids(0), shift_for_edge_voids(1), shift_for_edge_voids(2)]

    v = nearest_voids((0, 0, 0, 2))
    voids = {v[0]: [0, [0, 0, 0, 0, 1, 0], 0, 0], v[1]: [0, [0, 0, 0, 0, 0, 1], 0, 0]}
    """dictionary, its items are (x,y,z): [filled, [f0, f1, f2, f3, f4, f5], h, t]
    where (x,y,z) is a cube's center
    filled = 1 if all cube's faces are in complex
    f0 = 1 if the face number 0 in in complex amd so on
    h = 0 if the void is not in a hole and h = num_hole of the void is in a hole
    t is a time when the void was filled"""

    holes = {}  # dictionary containing all VOIDS that create holes
    total_holes = 0
    barcode = {}
    """dictionary, num_hole: [start_time, end_time]"""
    created_holes = []
    cubes_perimeter_edge = edge_voids((0, 0, 0, 2), shift_edge_voids)
    tags = []

    betti_2_total = 0
    betti_2_vector_changes = [0]
    betti_2_total_vector = [0]

    betti_1_total_vector = [0]

    per_2d = np.array([[len(perimeter)], [0], [0]])
    per_3d = np.array([[len(voids)], [0], [len(voids)]])

    betti_1_frequencies = {-1: [], 0: [], 1: [], 2: []}
    betti_2_frequencies = {0: [], 1: []}
    skipped = 0
    euler_char_prev = 1

    for i in tqdm(range(1, t)):
        perimeter_len = perimeter_len + [len(perimeter)]
        x = random.randint(0, len(perimeter) - 1)
        tile_selected = perimeter[x]
        # process += [return_vertices(tile_selected, shift_for_vertices)]
        perimeter.pop(x)
        eden[tile_selected][0] = 1

        v = nearest_voids(tile_selected)
        c = nearest_cubes(tile_selected)
        faces = update_void_dict(v, c, eden)

        t0, t1 = 0, 0
        if int(sum(faces[0]) / 6) == 1:
            t0 = i
        if int(sum(faces[1]) / 6) == 1:
            t1 = i
        if v[0] not in voids:
            voids[v[0]] = [int(sum(faces[0]) / 6), faces[0], 0, t0]
        else:
            voids[v[0]][0:2] = [int(sum(faces[0]) / 6), faces[0]]
            voids[v[0]][3] = t0
        if v[1] not in voids:
            voids[v[1]] = [int(sum(faces[1]) / 6), faces[1], 0, t1]
        else:
            voids[v[1]][0:2] = [int(sum(faces[1]) / 6), faces[1]]
            voids[v[1]][3] = t1

        eden, perimeter, nearest_n, nearest_neighbour_tiles = actualize_neighbors(tile_selected, eden, perimeter,
                                                                                  shift_neighbours)
        nearest_diag, nearest_diag_tiles = neighbours_diag(tile_selected, eden, shift_diag_neighbours)
        vertices, edges = actualize_vef(vertices, edges, nearest_n, nearest_diag)

        euler_character = euler_characteristic(vertices, edges, i - skipped + 1)

        if (model == 'no_betti_2_new' and euler_character <= euler_char_prev) or (model != 'no_betti_2_new'):
            betti_2, total_holes, eden, holes, voids, barcode, created_holes, tags, inner_perimeter = \
                increment_betti_2(eden, tile_selected, voids, total_holes, holes, barcode, i,
                                  created_holes, tags, inner_perimeter, perimeter, model)

        if (euler_character > euler_char_prev and model == 'no_betti_2_new') or (betti_2 == 1 and model == 'no_betti_2'):
            skipped += 1
            perimeter_len.pop()
            perimeter += [tile_selected]
            eden[tile_selected][0] = 0

            v = nearest_voids(tile_selected)
            c = nearest_cubes(tile_selected)
            faces = update_void_dict(v, c, eden)

            t0, t1 = 0, 0
            if int(sum(faces[0]) / 6) == 1:
                t0 = i
            if int(sum(faces[1]) / 6) == 1:
                t1 = i
            if v[0] not in voids:
                voids[v[0]] = [int(sum(faces[0]) / 6), faces[0], 0, t0]
            else:
                voids[v[0]][0:2] = [int(sum(faces[0]) / 6), faces[0]]
                voids[v[0]][3] = t0
            if v[1] not in voids:
                voids[v[1]] = [int(sum(faces[1]) / 6), faces[1], 0, t1]
            else:
                voids[v[1]][0:2] = [int(sum(faces[1]) / 6), faces[1]]
                voids[v[1]][3] = t1
            continue

        euler_char_prev = euler_character

        # update betti_1
        betti_1_total = return_betti_1(betti_2_total, euler_character)
        betti_1_total_vector += [betti_1_total]

        new_cubes = edge_voids(tile_selected, shift_edge_voids)
        for cube in new_cubes:
            if cube not in cubes_perimeter_edge:
                cubes_perimeter_edge += [cube]
            if cube in voids:
                if voids[cube][0] == 1.:
                    cubes_perimeter_edge.remove(cube)

        # update 3d perimeter
        total = len(voids)
        inner = len([x for x in voids if voids[x][2] != 0])
        outer = len([x for x in voids if voids[x][2] == 0])
        perim_3d = np.array([[total], [inner], [outer]])
        per_3d = np.c_[per_3d, perim_3d]

        # update 2d perimeter
        total = len(perimeter)
        inner = len(inner_perimeter)
        outer = total - inner
        perim_2d = np.array([[total], [inner], [outer]])
        per_2d = np.c_[per_2d, perim_2d]

        betti_2_vector_changes += [betti_2]
        betti_2_total += betti_2
        betti_2_total_vector += [betti_2_total]

        # update betti_1 fr
        betti_1_vector_changes = [betti_1_total_vector[j+1] - betti_1_total_vector[j] for j in range(i - skipped - 2*length - 1)]
        counter = collections.Counter(betti_1_vector_changes)
        a = [-1, 0, 1, 2]
        for k in a:
            betti_1_frequencies[k].append(counter[k]/i)

        # update betti_2 fr
        counter = collections.Counter(betti_2_vector_changes)
        a = [0, 1]
        for k in a:
            betti_2_frequencies[k].append(counter[k]/i)

    # perimeter_len = perimeter_len + [len(perimeter)]
    # final_barcode = barcode_forest(barcode, tags)
    # final_barcode.sort()

    # return eden, perimeter, process, perimeter_len, betti_2_vector_changes, betti_2_total, betti_2_total_vector, barcode,\
    #        betti_1_total, betti_1_total_vector, created_holes, holes, cubes_perimeter_edge, voids, tags, final_barcode,\
    #        per_3d, betti_1_frequencies, betti_2_frequencies
    # return per_3d, per_2d
    return eden, holes, betti_2_total_vector, voids, betti_1_total_vector, tile_selected


def increment_betti_2(eden, tile_selected, voids, total_holes, holes, barcode, time, created_holes, tags,
                      inner_perimeter, perimeter, model):  # , nearest_n, nearest_n_tiles):
    """betti_2 can increase only"""
    tile = np.array(tile_selected)
    v = nearest_voids(tile)

    if v[0] in holes_voids:
        num_hole = [x for x in holes if v[0] in holes[x]][0]
        per = 0
    else:
        per = 1
    """per = 0 means we are inside the hole"""

    num_possible_components = 2
    bfs = nearest_voids(tile)
    bfs = [[bfs[0]], [bfs[1]]]

    iterations = 0
    finished = [0] * num_possible_components
    merged = False
    while sum(finished) < num_possible_components - per and not merged:
        for j in range(num_possible_components):
            if finished[j] == 0:
                bfs, merged, finished = add_neighbours_bfs(bfs, j, iterations, merged, finished, eden, voids)
                if (iterations + 1) == len(bfs[j]):  # the hole is filled
                    finished[j] = 1
        iterations += 1

    betti_2 = 1 - int(merged)

    if model == 'no_betti_2':
        if betti_2 == 1:
            return betti_2, total_holes, eden, holes, voids, barcode, created_holes, tags, inner_perimeter

    # if betti_2 == 0 literally nothing happens
    if betti_2 == 1:
        if per == 0:

            if barcode[num_hole][0] == 0:
                tags = tags + [num_hole]
                barcode[num_hole][0] = 1

            holes.pop(num_hole)
            barcode[num_hole][1][1] = time + 1

            for i in range(num_possible_components):
                if finished[i] == 1:
                    total_holes += 1
                    holes[total_holes] = bfs[i].copy()
                    for void in bfs[i]:
                        if void not in holes_voids:
                            holes_voids += [void]
                        if void in voids:
                            voids[void][2] = total_holes
                        else:
                            voids[void] = [0, [0, 0, 0, 0, 0, 0], total_holes, 0]
                    barcode[total_holes] = [1, [time + 1, 0], barcode[num_hole][2] + [total_holes]]
                    created_holes = created_holes + [[barcode[total_holes], bfs[i].copy(), len(bfs[i]), total_holes]]
        else:
            for i in range(num_possible_components):
                if finished[i] == 1:
                    total_holes += 1
                    holes[total_holes] = bfs[i].copy()
                    for void in bfs[i]:
                        if void not in holes_voids:
                            holes_voids += [void]
                        if void in voids:
                            voids[void][2] = total_holes
                        else:
                            voids[void] = [0, [0, 0, 0, 0, 0, 0], total_holes, 0]
                        barcode[total_holes] = [0, [time + 1, 0], [total_holes]]
                    created_holes = created_holes + [[barcode[total_holes], bfs[i].copy(), len(bfs[i]), total_holes]]

    # change 2d inner perimeter
    if per == 0:  # we are inside a hole:
        inner_perimeter.remove(tile_selected)
        eden[tile_selected][2] = 0
        tile = list(tile_selected)
        if betti_2 == 0:
            hole = holes[num_hole]
            tiles = tiles_from_voids(hole)
            nearest_tiles = shift_for_neighbors(int(tile[3])) + (tile[:3] + [0])
            for n in nearest_tiles:
                n[3] = dimension(n[:3])
            nearest_tiles = [tuple(n) for n in nearest_tiles]
            for x in nearest_tiles:
                if x in perimeter and x in tiles and x not in inner_perimeter:
                    inner_perimeter.append(x)
                    eden[x][2] = 1
    else:
        if betti_2 == 1:
            hole = holes[total_holes]
            if len(hole) > 1:
                tiles = tiles_from_voids(hole)
                for x in tiles:
                    if x in perimeter:
                        inner_perimeter.append(x)
                        eden[x][2] = 1

    return betti_2, total_holes, eden, holes, voids, barcode, created_holes, tags, inner_perimeter, holes_voids


def add_neighbours_bfs(bfs, j, iterations, merged, finished, eden, voids):
    void_selected = bfs[j][iterations]
    if void_selected in voids:
        faces = voids[void_selected][1]
    else:
        faces = [0] * 6
    for i, face in enumerate(faces):
        if face == 0:
            direction = int(i / 2)
            if i % 2 == 1:
                sign = 1
            else:
                sign = -1
            shift = [0., 0., 0.]
            shift[direction] = sign
            new_void = tuple(np.array(void_selected) + shift)
            if new_void not in bfs[j]:
                bfs[j] += [new_void]
            # update merge/finish
            if new_void in bfs[1 - j]:  # if j new_void in the second part of gas
                merged = True
                finished[j] = 1
        if merged:
            break
    return bfs, merged, finished



# result = grow_eden()
Time = 20000
Model = 'standard'
Betti_1_frequencies, Betti_2_frequencies, Betti_1_total_vector, p2d, p3d, Betti_2_total_vector, Eden, Process, Created_holes, Holes = grow_eden(Time, Model, 10)

N = 00000
a = np.array([Betti_1_total_vector, Betti_2_total_vector, p2d[0], p3d[0]])
b = np.transpose(a)
np.savetxt(str(Time)+".csv", b, delimiter=",", header="b1, b2, p2, p3")

if not os.path.exists('pictures/'+str(int(Time/1000))+'k/'):
    os.makedirs('pictures/'+str(int(Time/1000))+'k/')

Tricube, Tricube_f, Tetracube, Tetracube_f = num_holes(Created_holes, Holes)
draw_tri_tetra(Tricube, Tricube_f, Tetracube, Tetracube_f, Time)
draw_diagram_holes(Created_holes, Holes, Time)
plot_b_per(Betti_1_total_vector, Betti_2_total_vector, p2d[0], p3d[0], Time, N)

# print(Created_holes[:3])
# max = 0
# hole = Created_holes[0][3]
# for x in Created_holes:
#     if x[2] > max and x[0][1] == 1:
#         max = x[2]
#         num_hole = x[3]

print('a')











# Time = 100
# Model = 'standard'
# Betti_1_frequencies, Betti_2_frequencies, Betti_1_total_vector, p2d, p3d, Betti_2_total_vector, Eden, Process, Created_holes, Holes = grow_eden(Time, Model, 10)
# a = 10
# print(Eden)
# Eden = {(0,0,0.5,2): [1]}
# list_cubes = [nearest_cubes(x) for x in Eden if Eden[x][0] == 1]
# tiles = [x for x in Eden if Eden[x][0] == 1]
# list_cubes = list(itertools.chain.from_iterable(list_cubes))
# list_cubes = list(itertools.chain.from_iterable(list_cubes))
# list_cubes = list(dict.fromkeys(list_cubes))
# draw_complex(list_cubes, 0, tiles)
# draw_eden(Eden, Time)
# Eden, Perimeter, Process, Perimeter_len, Betti_2_vector_changes, Betti_2, Betti_2_total_vector, Barcode, Betti_1_total, \
#     Betti_1_total_vector, Created_Holes, Holes, Cubes_perimeter_edge, Voids, Tags, Final_barcode, Per_3d, \
#     Betti_1_frequencies, Betti_2_frequencies = grow_eden(Time)
# print(Eden)
# draw_complex(Eden, Time, last_tile)
# f1 = open("per_3d.txt", "w+")

# for q in range(1):
#     P3, P2 = grow_eden(Time)
# f1.close()
# draw_barcode(Final_barcode, Time)
# draw_eden(Eden, Time)


"""draw a 3d hole"""
# a = nearest_cubes((0, 0, 0, 2))
# b = nearest_cubes((0, 0.5, 0.5, 1))
# c = nearest_cubes((0.0, 1.5, 0.5, 1.0))
# x = np.concatenate(np.concatenate((a, b, c)))
# x_ = np.unique(x, axis=0)
# x_ = np.delete(x_, [7, 13, 10], axis=0)
# draw_complex(x_, 0, None)




