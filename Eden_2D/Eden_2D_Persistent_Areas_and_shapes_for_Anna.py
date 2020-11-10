#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
from tqdm import tqdm
from Drawing import draw_barcode, draw_polyomino, draw_polyomino_holes
from Grow_Eden_Supplementary_Functions import start_eden_2d, actualize_neighbors, neighbours_diag, actualize_vef, \
    increment_betti_1_euler, increment_betti_1, barcode_forest, num_holes
from Plots import plot_b_per, draw_diagram_holes, draw_tri_tetra
import os
"""
Created on Thu Feb 13 19:04:05 2020
 Hi Anna, with this script you can:
     generate 2 dimensional ECG processes
     generate images of the process
     keep track of all the holes (there is only one kind in 2D)
     analize PH 
     
     I suggest that we go together throw the code. And please, 
     feel free to make/suggest any changes and comment the code.
     
     Remember, the aim is to generate in 3D adding 2 cells.
     Once you understand this code I will send you the one 
     in 3D adding 3D cubes if you want me to. But perhaps you would prefer to
     implement it from zero to have more fun. I am OK with both scenarios
     and would be happy to talk about the algorithm in any case that you choose.
@author: Erika Roldan Roa
"""
"""
Created on Thu Nov 21 19:34:33 2019
@author: err
"""


def grow_eden(t):
    vertices = 4
    edges = 4

    eden, perimeter = start_eden_2d()  # perimeter is an array consisting of all tiles that are on the perimeter
    process = [(0, 0)]  # an array consisting of all tiles that were added

    perimeter_len = []  # an array consisting of perimeter lengths on every time step
    holes = {}
    total_holes = 0
    barcode = {}
    created_holes = []
    tags = []

    betti_1_total = 0
    betti_1_vector_changes = [0]
    betti_1_total_vector = [0]

    for i in tqdm(range(1, t)):
        perimeter_len = perimeter_len + [len(perimeter)]
        x = random.randint(0, len(perimeter) - 1)

        tile_selected = perimeter[x]
        process = process + [tile_selected]

        perimeter.pop(x)
        eden[tile_selected][0] = 1

        eden, perimeter, nearest_n, nearest_neighbour_tiles = actualize_neighbors(tile_selected, eden, perimeter)
        nearest_diag_tiles = neighbours_diag(eden, tile_selected)

        vertices, edges = actualize_vef(vertices, edges, nearest_n, nearest_diag_tiles)
        betti_1, total_holes, eden, barcode, holes, created_holes, tags = increment_betti_1(eden, tile_selected,
                                                                                            nearest_n,
                                                                                            nearest_neighbour_tiles, barcode, i,
                                                                                            holes, total_holes,
                                                                                            created_holes, tags)
        # print('betti_1: ', betti_1)
        # draw_polyomino(eden, 0)
        betti_1_vector_changes = betti_1_vector_changes + [betti_1]
        betti_1_total += betti_1
        betti_1_total_vector = betti_1_total_vector + [betti_1_total]

        final_barcode = barcode_forest(barcode, tags)

    l = len(perimeter)
    perimeter_len = perimeter_len + [l]

    return eden, perimeter, betti_1_total_vector, betti_1_vector_changes, barcode, holes, betti_1_total, \
           created_holes, process, perimeter_len, tags, final_barcode


def grow_eden_debugging(t, ordered_tiles):
    vertices = 4
    edges = 4

    eden, perimeter = start_eden_2d()

    holes = {}
    total_holes = 0
    barcode = {}
    created_holes = []
    tags = []

    betti_1_total = 0
    betti_1_vector = []
    betti_1_euler_total = 0
    betti_1_total_vector = [0]
    len_perimeter = [4]

    for i in tqdm(range(1, t)):
        tile_selected = ordered_tiles[i]
        perimeter.remove(tile_selected)
        eden[tile_selected][0] = 1

        eden, perimeter, nearest_n, nearest_n_tiles = actualize_neighbors(tile_selected, eden, perimeter)
        n = neighbours_diag(eden, tile_selected)

        vertices, edges = actualize_vef(vertices, edges, nearest_n, n)
        betti_1, total_holes, eden, barcode, holes, created_holes, tags = increment_betti_1(eden, tile_selected,
                                                                                            nearest_n,
                                                                                            nearest_n_tiles, barcode, i,
                                                                                            holes, total_holes,
                                                                                            created_holes, tags)
        betti_1_vector = betti_1_vector + [betti_1]
        betti_1_total = betti_1_total + betti_1
        betti_1_total_vector += [betti_1_total]

        betti_1_euler_total = increment_betti_1_euler(vertices, edges, i)
        if betti_1_total != betti_1_euler_total:
            raise ValueError('betti_1_total does not equal betti_1 with euler')
        len_perimeter += [len(perimeter)]

        final_barcode = barcode_forest(barcode, tags)

    return eden, perimeter, betti_1_vector, betti_1_total_vector, barcode, holes, betti_1_total, betti_1_euler_total, created_holes, tags, final_barcode, len_perimeter

def read_eden_txt(filename):
    eden = []
    for t in open(filename).read().split('), ('):
        # print(t)
        a, b, c = t.strip('()[]').split(',')
        a = a.strip()
        b = b.strip(')')
        c = c.strip(")]\n")
        eden.append(((int(a), int(b)), float(c)))
    return eden


#####################
# with open('3000_3D_barcode_2_24.txt','w') as f:
#     #     f.writelines( '%s %s\n' % tuple(tu) for tu in final_barcode )

Eden_f = read_eden_txt("sample_time_list.txt")
Eden = [x[0] for x in Eden_f]
Time = len(Eden)
Eden, Perimeter, Betti_1_vector, Betti_1_total_vector, Barcode, Holes, Betti_1_total, Betti_1_euler_total, \
    Created_holes, Tags, Final_barcode, Len_perimeter = grow_eden_debugging(len(Eden), Eden)


if not os.path.exists('pictures/'+str(int(Time/1000))+'k/'):
    os.makedirs('pictures/'+str(int(Time/1000))+'k/')
# plot_b_per(Betti_1_total_vector, Len_perimeter, Time)
# draw_diagram_holes(Created_holes, Holes, Time)
Tromino, Tromino_f, Tetromino, Tetromino_f = num_holes(Created_holes, Holes)
draw_tri_tetra(Tromino, Tromino_f, Tetromino, Tetromino_f, Time)

print('a')
# Time = 5000
# Eden, Perimeter, Betti_1_total_vector, Betti_1_vector_changes, Barcode, Holes, Betti_1_total, Created_holes, Process, \
#     Perimeter_len, Tags, a = grow_eden(Time)
# draw_polyomino(Eden, Time)
# print('betti_1: ', Betti_1_total)
# print('perimeter: ', Perimeter_len[-1])
# eden, perimeter, betti_1_vector, betti_1_euler_total_vector, barcode, holes, betti_1_total, betti_1_euler_total, created_holes, tags, final_barcode\
#     = grow_eden_debugging(Time, Process)
# print(betti_1_total, betti_1_euler_total)

