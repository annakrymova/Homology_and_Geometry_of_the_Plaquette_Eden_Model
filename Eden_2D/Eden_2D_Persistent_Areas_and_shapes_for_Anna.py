#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
from tqdm import tqdm
from Eden_2D.Drawing import draw_barcode, draw_polyomino, draw_polyomino_holes
from Eden_2D.Grow_Eden_Supplementary_Functions import start_eden_2d, actualize_neighbors, neighbours, actualize_vef, \
    increment_betti_1, increment_betti_2, barcode_forest

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

    eden, perimeter = start_eden_2d()
    process = [(0, 0)]

    perimeter_len = []
    holes = {}
    total_holes = 0
    barcode = {}
    created_holes = []
    tags = []

    betti_1_total = 0
    betti_1_vector_changes = [0]
    betti_1_total_vector = [0]

    for i in tqdm(range(1, t)):
        l = len(perimeter)
        perimeter_len = perimeter_len + [l]
        x = random.randint(0, l - 1)

        tile_selected = perimeter[x]
        process = process + [tile_selected]

        perimeter.pop(x)
        eden[tile_selected][0] = 1

        eden, perimeter, nearest_n, nearest_n_tiles = actualize_neighbors(tile_selected, eden, perimeter)
        n = neighbours(eden, tile_selected)

        vertices, edges = actualize_vef(vertices, edges, nearest_n, n)
        betti_1, total_holes, eden, barcode, holes, created_holes, tags = increment_betti_2(eden, tile_selected,
                                                                                            nearest_n,
                                                                                            nearest_n_tiles, barcode, i,
                                                                                            holes, total_holes,
                                                                                            created_holes, tags)
        betti_1_vector_changes = betti_1_vector_changes + [betti_1]
        betti_1_total = betti_1_total + betti_1
        betti_1_total_vector = betti_1_total_vector + [betti_1_total]

        # final_barcode = barcode_forest(barcode, tags)

    l = len(perimeter)
    perimeter_len = perimeter_len + [l]

    return eden, perimeter, betti_1_total_vector, betti_1_vector_changes, barcode, holes, betti_1_total, \
           created_holes, process, perimeter_len  # , tags, final_barcode


def grow_eden_debugging(t, ordered_tiles):
    vertices = 4
    edges = 4

    eden, perimeter = start_eden_2d()
    process = [(0, 0)]

    holes = {}
    total_holes = 0
    barcode = {}
    created_holes = []
    tags = []

    betti_2_total = 0
    betti_2_vector = []
    betti_1_total = 0
    betti_1_total_vector = []

    for i in range(1, t):
        tile_selected = ordered_tiles[i]
        perimeter.remove(tile_selected)
        eden[tile_selected][0] = 1

        eden, perimeter, nearest_n, nearest_n_tiles = actualize_neighbors(tile_selected, eden, perimeter)
        n = neighbours(eden, tile_selected)

        vertices, edges = actualize_vef(vertices, edges, nearest_n, n)
        betti_2, total_holes, eden, barcode, holes, created_holes, tags = increment_betti_2(eden, tile_selected,
                                                                                            nearest_n,
                                                                                            nearest_n_tiles, barcode, i,
                                                                                            holes, total_holes,
                                                                                            created_holes, tags)
        betti_2_vector = betti_2_vector + [betti_2]
        betti_2_total = betti_2_total + betti_2

        betti_1_total_vector = increment_betti_1(vertices, edges, i, betti_2_total)

        final_barcode = barcode_forest(barcode, tags)

    return eden, perimeter, betti_2_vector, betti_1_total_vector, barcode, holes, betti_2_total, betti_1_total, created_holes, tags, final_barcode


#####################
# with open('3000_3D_barcode_2_24.txt','w') as f:
#     #     f.writelines( '%s %s\n' % tuple(tu) for tu in final_barcode )


Time = 1000
Eden, Perimeter, Betti_1_total_vector, Betti_1_vector_changes, Barcode, Holes, Betti_1_total, Created_holes, Process, Perimeter_len = grow_eden(Time)
# draw_polyomino(Eden, Time)
print('betti_1: ', Betti_1_total)
print('perimeter: ', Perimeter_len[-1])

