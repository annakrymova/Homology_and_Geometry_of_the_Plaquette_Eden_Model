from tqdm import tqdm
import random
import numpy as np
from Supplementary_functions import start_eden_2d_in_3d, actualize_neighbors, neighbours_diag, actualize_vef, \
    nearest_cubes, \
    nearest_voids, dimension, shift_for_neighbors, shift_for_neighbours_diag, check_cube_in_eden, update_void_dict

from Drawing import draw_eden, draw_complex, draw_square
import sys


def grow_eden(t):
    vertices = 4
    edges = 4

    eden, perimeter = start_eden_2d_in_3d()  # perimeter is an array consisting of all tiles that are on the perimeter
    process = [(0, 0, 0, 2)]  # an array consisting of all tiles that were added

    perimeter_len = []  # an array consisting of perimeter lengths at every time step

    shift_neighbours = [shift_for_neighbors(0), shift_for_neighbors(1), shift_for_neighbors(2)]
    shift_diag_neighbours = [shift_for_neighbours_diag(0), shift_for_neighbours_diag(1), shift_for_neighbours_diag(2)]

    v = nearest_voids(process[0])
    voids = {v[0]: [0, [0, 0, 0, 0, 1, 0], 0], v[1]: [0, [0, 0, 0, 0, 0, 1], 0]}
    """dictionary, its items are (x,y,z): [filled, [f0, f1, f2, f3, f4, f5]]
    where (x,y,z) is cube's center
    filled = 1 if all cube's faces are in complex
    f0 = 1 if the face number 0 in in complex amd so on """

    holes = {}  # dictionary containing all VOIDS that create holes
    total_holes = 0
    barcode = {}
    created_holes = []
    tags = []

    betti_2_total = 0
    betti_2_vector_changes = [0]
    betti_2_total_vector = [0]

    for i in tqdm(range(1, t)):
        perimeter_len = perimeter_len + [len(perimeter)]
        x = random.randint(0, len(perimeter) - 1)
        tile_selected = perimeter[x]
        process = process + [tile_selected]
        perimeter.pop(x)
        eden[tile_selected][0] = 1

        v = nearest_voids(tile_selected)
        c = nearest_cubes(tile_selected)
        faces = update_void_dict(v, c, eden)
        # print(faces)
        voids[v[0]] = [int(sum(faces[0]) / 6), faces[0], 0]
        voids[v[1]] = [int(sum(faces[1]) / 6), faces[1], 0]

        # check that voids dictionary corresponds to real complex
        total_faces = sum(np.array(list(voids.values()))[:, 1].sum())
        if total_faces != 2 * len(process):
            sys.exit('Something wrong with void function')
        eden, perimeter, nearest_n, nearest_neighbour_tiles = actualize_neighbors(tile_selected, eden, perimeter, shift_neighbours)
        nearest_diag, nearest_diag_tiles = neighbours_diag(tile_selected, eden, shift_diag_neighbours)
        vertices, edges = actualize_vef(vertices, edges, nearest_n, nearest_diag)

        betti_2, total_holes, eden, holes, voids = increment_betti_2(eden, tile_selected, voids, total_holes, holes)
        
        betti_2_vector_changes = betti_2_vector_changes + [betti_2]
        betti_2_total = betti_2_total + betti_2

    perimeter_len = perimeter_len + [len(perimeter)]

    return eden, perimeter, process, perimeter_len, betti_2_vector_changes, betti_2_total  # , tags, final_barcode


def increment_betti_2(eden, tile_selected, voids, total_holes, holes):  # , nearest_n, nearest_n_tiles):
    """betti_2 can increase only"""
    tile = np.array(tile_selected)
    #  todo: probably we can delete the hole variable from eden dictionary and add it to void dictionary.
    #   or leave it in both dictionaries. THINK ABOUT IT!
    if eden[tile_selected][2] == 0:
        per = 1  # This is 1 if the tile added doesn't create a hole
    else:
        num_hole = eden[tile_selected][2]
        per = 0
    betti_2 = 0
    cube_1, cube_2 = nearest_cubes(tile)
    c_1 = check_cube_in_eden(cube_1, eden)
    c_2 = check_cube_in_eden(cube_2, eden)
    if c_1 or c_2:
        betti_2 += 1
        draw_eden(eden, 0, tile_selected)
        a = 5

    num_possible_components = 2
    bfs = nearest_voids(tile)
    bfs = [[bfs[0]], [bfs[1]]]

    iterations = 0
    finished = [0] * num_possible_components
    merged = finished.copy()
    while sum(finished) < num_possible_components - per:
        for j in range(0, num_possible_components):
            if finished[j] == 0:
                bfs, merged, finished = add_neighbours_bfs(bfs, j, iterations, merged, finished, eden, voids)
                if (iterations + 1) == len(bfs[j]):  # the hole is filled
                    finished[j] = 1
            iterations = iterations + 1

    return betti_2


def add_neighbours_bfs(bfs, j, iterations, merged, finished, eden, voids):
    void_selected = bfs[j][iterations]
    faces = voids[void_selected][1]
    for i, face in enumerate(faces):
        if face == 0:
            direction = int(i/2)
            if i % 2 == 1:
                sign = 1
            else:
                sign = -1
            shift = [0., 0., 0.]
            shift[direction] = sign
            new_void = tuple(np.array(void_selected)+shift)
            bfs[j] += [new_void]
            # update merge/finish
            if new_void in bfs[1-j]:  # if j new_void in the second part of gas
                if 1-j < j:
                    merged[j] = 1
                    finished[j] = 1
                if 1-j > j:
                    merged[1-j] = 1
                    finished[1-j] = 1
    return bfs, merged, finished


c1, c2 = nearest_cubes((0, 0, 0, 2))
# draw(c1+c2, 0)
# nearest_voids((0, 0, 0, 2))
Time = 1000
Eden, Perimeter, Process, Perimeter_len = grow_eden(Time)
# Nearest_diag, tiles = neighbours_diag([0, 0, 0, 2], Eden)
# increment_betti_2(Eden, (0, 0, 0, 2))
draw_eden(Eden, Time, '')
print('hi')





