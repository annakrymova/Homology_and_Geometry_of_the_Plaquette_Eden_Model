from tqdm import tqdm
import random
import numpy as np
from Supplementary_functions import start_eden_2d_in_3d, actualize_neighbors, neighbours_diag, actualize_vef, nearest_cubes, \
                             nearest_voids, dimension, shift_for_neighbors, shift_for_neighbours_diag, check_cube_in_eden, update_void_dict

from Drawing import draw, draw_square


def grow_eden(t):
    vertices = 4
    edges = 4

    eden, perimeter = start_eden_2d_in_3d()  # perimeter is an array consisting of all tiles that are on the perimeter
    process = [(0, 0, 0, 2)]  # an array consisting of all tiles that were added

    perimeter_len = []  # an array consisting of perimeter lengths at every time step

    shift_neighbours = [shift_for_neighbors(0), shift_for_neighbors(1), shift_for_neighbors(2)]
    shift_diag_neighbours = [shift_for_neighbours_diag(0), shift_for_neighbours_diag(1), shift_for_neighbours_diag(2)]

    v = nearest_voids(process[0])
    c = nearest_cubes(process[0])
    voids = {v[0]: [0, [0, 0, 0, 0, 1, 0]], v[1]: [0, [0, 0, 0, 0, 0, 1]]}

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

        v = nearest_voids(tile_selected)
        c = nearest_cubes(tile_selected)
        faces = update_void_dict(v, c, eden)
        voids[v[0]] = [int(sum(faces[0])/6), faces[0]]
        voids[v[1]] = [int(sum(faces[1])/6), faces[1]]

        eden, perimeter, nearest_n, nearest_neighbour_tiles = actualize_neighbors(tile_selected, eden, perimeter, shift_neighbours)
        nearest_diag, nearest_diag_tiles = neighbours_diag(tile_selected, eden, shift_diag_neighbours)
        vertices, edges = actualize_vef(vertices, edges, nearest_n, nearest_diag)
        # b = increment_betti_2(eden, tile_selected)

    perimeter_len = perimeter_len + [len(perimeter)]

    return eden, perimeter, process, perimeter_len  # , tags, final_barcode


def increment_betti_2(eden, tile_selected):  # , nearest_n, nearest_n_tiles):
    """betti_2 can increase only"""
    tile = np.array(tile_selected)
    if eden[tile_selected][2] == 0:
        per = 1  # This is 1 if the tile added was in the out perimeter
    else:
        num_hole = eden[tile_selected][2]
        per = 0
    betti_2 = 0
    cube_1, cube_2 = nearest_cubes(tile)
    c_1 = check_cube_in_eden(cube_1, eden)
    c_2 = check_cube_in_eden(cube_2, eden)
    if c_1 or c_2:
        betti_2 += 1
        draw(eden, 0, tile_selected)
    else:
        num_possible_components = 2
        bds = nearest_voids(tile)
        iterations = 0

    return betti_2


c1, c2 = nearest_cubes((0, 0, 0, 2))
# draw(c1+c2, 0)
# nearest_voids((0, 0, 0, 2))
Time = 1000
Eden, Perimeter, Process, Perimeter_len = grow_eden(Time)
# Nearest_diag, tiles = neighbours_diag([0, 0, 0, 2], Eden)
# increment_betti_2(Eden, (0, 0, 0, 2))
draw(Eden, Time, '')
print('hi')





