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
import os
import itertools
import sys
from itertools import groupby
import collections
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection

# try:
#     from line_profiler import LineProfiler
#
#     def do_profile(follow=[]):
#         def inner(func):
#             def profiled_func(*args, **kwargs):
#                 try:
#                     profiler = LineProfiler()
#                     profiler.add_function(func)
#                     for f in follow:
#                         profiler.add_function(f)
#                     profiler.enable_by_count()
#                     return func(*args, **kwargs)
#                 finally:
#                     profiler.print_stats()
#             return profiled_func
#         return inner
#
# except ImportError:
#     def do_profile(follow=[]):
#         "Helpful if you accidentally leave in production!"
#         def inner(func):
#             def nothing(*args, **kwargs):
#                 return func(*args, **kwargs)
#             return nothing
#         return inner


def increment_betti_2(eden, tile_selected, voids, total_holes, holes, barcode, time, created_holes, tags,
                      inner_perimeter, perimeter, model, holes_voids):  # , nearest_n, nearest_n_tiles):
    # betti_2 = 0
    # return betti_2, total_holes, eden, holes, voids, barcode, created_holes, tags, inner_perimeter, holes_voids
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




