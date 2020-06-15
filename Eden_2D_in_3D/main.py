from tqdm import tqdm
import random
import numpy as np
from Supplementary_functions import start_eden_2d_in_3d, actualize_neighbors, neighbours_diag, actualize_vef, \
    nearest_cubes, nearest_voids, dimension, shift_for_neighbors, shift_for_neighbours_diag, \
    check_cube_in_eden, update_void_dict, shift_vertices, return_vertices, euler_characteristic, \
    return_betti_1, edge_voids, shift_for_edge_voids, barcode_forest

from Drawing import draw_eden, draw_complex, draw_square, draw_barcode
import sys


def grow_eden(t):
    vertices = 4
    edges = 4

    eden, perimeter = start_eden_2d_in_3d()  # perimeter is an array consisting of all tiles that are on the perimeter

    shift_for_vertices = shift_vertices(0), shift_vertices(1), shift_vertices(2)
    process = [return_vertices((0, 0, 0, 2), shift_for_vertices)]  # an array consisting of all tiles that were added (specified by there 4 vertices)

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

    for i in tqdm(range(1, t)):
        perimeter_len = perimeter_len + [len(perimeter)]
        x = random.randint(0, len(perimeter) - 1)
        tile_selected = perimeter[x]

        process += [return_vertices(tile_selected, shift_for_vertices)]
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

        new_cubes = edge_voids(tile_selected, shift_edge_voids)
        for cube in new_cubes:
            if cube not in cubes_perimeter_edge:
                cubes_perimeter_edge += [cube]
            if cube in voids:
                if voids[cube][0] == 1.:
                    cubes_perimeter_edge.remove(cube)

        # check that voids dictionary corresponds to real complex
        total_faces = sum(np.array(list(voids.values()))[:, 1].sum())
        # if total_faces != 2 * len(process):
        #     sys.exit('Something wrong with void function')
        eden, perimeter, nearest_n, nearest_neighbour_tiles = actualize_neighbors(tile_selected, eden, perimeter,
                                                                                  shift_neighbours)
        nearest_diag, nearest_diag_tiles = neighbours_diag(tile_selected, eden, shift_diag_neighbours)
        vertices, edges = actualize_vef(vertices, edges, nearest_n, nearest_diag)

        betti_2, total_holes, eden, holes, voids, barcode, created_holes, tags = increment_betti_2(eden, tile_selected, voids,
                                                                                                   total_holes, holes, barcode,
                                                                                                   i, created_holes, tags)
        betti_2_vector_changes += [betti_2]
        betti_2_total += + betti_2
        betti_2_total_vector += [betti_2_total]

        euler_character = euler_characteristic(vertices, edges, i + 1)
        betti_1_total = return_betti_1(betti_2_total, euler_character)
        betti_1_total_vector += [betti_1_total]

    perimeter_len = perimeter_len + [len(perimeter)]
    final_barcode = barcode_forest(barcode, tags)
    final_barcode.sort()

    return eden, perimeter, process, perimeter_len, betti_2_vector_changes, betti_2_total, betti_2_total_vector, barcode,\
           betti_1_total, betti_1_total_vector, created_holes, holes, cubes_perimeter_edge, voids, tags, final_barcode


def increment_betti_2(eden, tile_selected, voids, total_holes, holes, barcode, time, created_holes, tags):  # , nearest_n, nearest_n_tiles):
    """betti_2 can increase only"""
    tile = np.array(tile_selected)
    holes_voids = [v for v in voids if voids[v][2] != 0]
    v = nearest_voids(tile)

    if v[0] in holes_voids:
        per = 0
        num_hole = [x for x in holes if v[0] in holes[x]][0]
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
                        if void in voids:
                            voids[void][2] = total_holes
                        else:
                            voids[void] = [0, [0, 0, 0, 0, 0, 0], total_holes, 0]
                    barcode[total_holes] = [1, [time + 1, 0], barcode[num_hole][2] + [total_holes]]
                    created_holes = created_holes + [[barcode[total_holes], bfs[i].copy(), len(bfs[i])]]
        else:
            for i in range(num_possible_components):
                if finished[i] == 1:
                    total_holes += 1
                    holes[total_holes] = bfs[i].copy()
                    for void in bfs[i]:
                        if void in voids:
                            voids[void][2] = total_holes
                        else:
                            voids[void] = [0, [0, 0, 0, 0, 0, 0], total_holes, 0]
                        barcode[total_holes] = [0, [time + 1, 0], [total_holes]]
                    created_holes = created_holes + [[barcode[total_holes], bfs[i].copy(), len(bfs[i])]]
    return betti_2, total_holes, eden, holes, voids, barcode, created_holes, tags


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


Time = 500
Eden, Perimeter, Process, Perimeter_len, Betti_2_vector_changes, Betti_2, Betti_2_total_vector, Barcode, Betti_1_total, \
    Betti_1_total_vector, Created_Holes, Holes, Cubes_perimeter_edge, Voids, Tags, Final_barcode = grow_eden(Time)
draw_barcode(Final_barcode, Time)
draw_eden(Eden, Time)




