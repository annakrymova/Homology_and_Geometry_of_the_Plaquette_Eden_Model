import numpy as np
import sys
from tqdm import tqdm
import random
import collections
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import collections
from scipy.optimize import curve_fit
import gudhi as gd

"""GROWING"""
def grow_eden(t, model):
    vertices = 4
    edges = 4

    eden, perimeter = start_eden_2d_in_3d()
    """perimeter is an array consisting of all tiles that are on the perimeter"""
    inner_perimeter = []
    shift_for_vertices = shift_vertices(0), shift_vertices(1), shift_vertices(2)
    process = [return_vertices((0, 0, 0.5, 2), shift_for_vertices)]
    """array for MAYA"""
    perimeter_len = []  # an array consisting of perimeter lengths at every time step

    shift_neighbours = [shift_for_neighbors(0), shift_for_neighbors(1), shift_for_neighbors(2)]
    shift_diag_neighbours = [shift_for_neighbours_diag(0), shift_for_neighbours_diag(1), shift_for_neighbours_diag(2)]

    v = nearest_voids((0, 0, 0, 2))
    voids = {v[0]: [0, [0, 0, 0, 0, 1, 0], 0, 0], v[1]: [0, [0, 0, 0, 0, 0, 1], 0, 0, 0]}
    """dictionary, its items are (x,y,z): [filled, [f0, f1, f2, f3, f4, f5], h, t, v]
    where (x,y,z) is a cube's center
    filled = 1 if all cube's faces are in complex
    f0 = 1 if the face number 0 in in complex amd so on
    h = 0 if the void is not in a hole and h = num_hole if the void is in a hole
    t is time when the void was filled
    v is time when the void created a hole of volume 1"""

    """holes is a dictionary containing all VOIDS that create holes"""
    holes = {}
    total_holes = 0
    """barcode is a dictionary, num_hole: [start_time, end_time]"""
    barcode = {}
    created_holes = []
    """list of holes that were divided"""
    tags = []

    betti_2_total = 0
    betti_2_vector_changes = [0]
    betti_2_total_vector = [0]

    betti_1_total_vector = [0]

    per_2d = [len(perimeter)]
    per_3d = [len(voids)]

    euler_char_prev = 1
    holes_voids = []

    skipped = 0
    n_filled_cubes = 0
    size = 1

    pbar = tqdm(total=t)
    pbar.update(1)

    while size < t:
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
            t0 = size
        if int(sum(faces[1]) / 6) == 1:
            t1 = size
        if v[0] not in voids:
            voids[v[0]] = [int(sum(faces[0]) / 6), faces[0], 0, t0, 0]
        else:
            voids[v[0]][0:2] = [int(sum(faces[0]) / 6), faces[0]]
            voids[v[0]][3] = t0
        if v[1] not in voids:
            voids[v[1]] = [int(sum(faces[1]) / 6), faces[1], 0, t1, 0]
        else:
            voids[v[1]][0:2] = [int(sum(faces[1]) / 6), faces[1]]
            voids[v[1]][3] = t1

        eden, perimeter, nearest_n, nearest_neighbour_tiles = actualize_neighbors(tile_selected, eden, perimeter,
                                                                                  shift_neighbours)
        nearest_diag, nearest_diag_tiles = neighbours_diag(tile_selected, eden, shift_diag_neighbours)
        vertices, edges = actualize_vef(vertices, edges, nearest_n, nearest_diag)

        euler_character = euler_characteristic(vertices, edges, size - skipped + 1, n_filled_cubes)

        if (model == 1 and euler_character <= euler_char_prev) or model == 0 or model == 2:
            betti_2, total_holes, eden, holes, voids, barcode, created_holes, tags, inner_perimeter, holes_voids,\
                n_filled_cubes = increment_betti_2(eden, tile_selected, voids, total_holes, holes, barcode, size,
                                                   created_holes, tags, inner_perimeter, perimeter, model, holes_voids, n_filled_cubes)

        if euler_character > euler_char_prev and model == 1:
            skipped += 1
            perimeter_len.pop()
            perimeter += [tile_selected]
            eden[tile_selected][0] = 0
            del process[-1]

            v = nearest_voids(tile_selected)
            c = nearest_cubes(tile_selected)
            faces = update_void_dict(v, c, eden)

            t0, t1 = 0, 0
            if int(sum(faces[0]) / 6) == 1:
                t0 = size
            if int(sum(faces[1]) / 6) == 1:
                t1 = size
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

        per_2d += [len(perimeter)]
        per_3d += [len(voids)]

        betti_2_vector_changes += [betti_2]
        betti_2_total += betti_2
        betti_2_total_vector += [betti_2_total]

        pbar.update(1)
        euler_char_prev = euler_character
        size += 1

        # update betti_1
        betti_1_total = return_betti_1(betti_2_total, euler_character)
        betti_1_total_vector += [betti_1_total]

    final_barcode = barcode_forest(barcode, tags)
    final_barcode.sort(reverse=True)

    return betti_1_total_vector, per_2d, per_3d, betti_2_total_vector, eden,\
           process, created_holes, holes, final_barcode

"""SUPPLEMENTARY FUNCTIONS"""
def increment_betti_2(eden, tile_selected, voids, total_holes, holes, barcode, time, created_holes, tags,
                      inner_perimeter, perimeter, model, holes_voids, n_filled_cubes):  # , nearest_n, nearest_n_tiles):
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
    filled = 0
    if betti_2 == 1:
        if per == 0:
            if barcode[num_hole][0] == 0:
                tags = tags + [num_hole]
                barcode[num_hole][0] = 1
            holes.pop(num_hole)
            barcode[num_hole][1][1] = time + 1

            for i in range(num_possible_components):
                if finished[i] == 1:
                    if len(bfs[i]) == 1 and model == 2:
                        voids[4] = time
                        filled += 1
                        n_filled_cubes += 1
                    else:
                        total_holes += 1
                        holes[total_holes] = bfs[i].copy()
                        for void in bfs[i]:
                            if void not in holes_voids:
                                holes_voids += [void]
                            if void in voids:
                                voids[void][2] = total_holes
                            else:
                                voids[void] = [0, [0, 0, 0, 0, 0, 0], total_holes, 0, 0]
                        barcode[total_holes] = [1, [time + 1, float('inf')], barcode[num_hole][2] + [total_holes]]
                        created_holes = created_holes + [[barcode[total_holes], bfs[i].copy(), len(bfs[i]), total_holes]]
        else:
            for i in range(num_possible_components):
                if finished[i] == 1:
                    """catching cubes of volume 1"""
                    if len(bfs[i]) == 1 and model == 2:
                        voids[4] = time
                        filled += 1
                        n_filled_cubes += 1
                    else:
                        total_holes += 1
                        holes[total_holes] = bfs[i].copy()
                        for void in bfs[i]:
                            if void not in holes_voids:
                                holes_voids += [void]
                            if void in voids:
                                voids[void][2] = total_holes
                            else:
                                voids[void] = [0, [0, 0, 0, 0, 0, 0], total_holes, 0]
                        barcode[total_holes] = [0, [time + 1, float('inf')], [total_holes]]
                        created_holes = created_holes + [[barcode[total_holes], bfs[i].copy(), len(bfs[i]), total_holes]]

    if betti_2 != 0:
        betti_2 -= filled

    # change 2d inner perimeter
    if per == 0:  # we are inside a hole:
        inner_perimeter.remove(tile_selected)
        eden[tile_selected][2] = 0
        tile = list(tile_selected)
        if betti_2 == 0 and filled == 0:
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

    return betti_2, total_holes, eden, holes, voids, barcode, created_holes, tags, inner_perimeter, holes_voids, n_filled_cubes

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

def return_frequencies_1(vect, time):
    changes = [vect[i+1]-vect[i] for i in range(len(vect)-1)]

    values = [-1, 0, 1, 2]
    freq = {i: [0] for i in values}

    for i in tqdm(range(1, time+1), position=0, leave=True):
        counter = collections.Counter(changes[:i])
        for k in values:
            freq[k].append(counter[k]/i)
    return freq, changes

def return_frequencies_2(vect, time):
    changes = [vect[i+1]-vect[i] for i in range(len(vect)-1)]
    values = [-1, 0, 1]
    freq = {i: [0] for i in values}

    for i in tqdm(range(1, time+1), position=0, leave=True):
        counter = collections.Counter(changes[:i])
        for k in values:
            freq[k].append(counter[k]/i)
    return freq, changes

def dimension(coordinates):
    if coordinates[0] % 1 == 0.5:
        d = 0
    elif coordinates[1] % 1 == 0.5:
        d = 1
    else:
        d = 2
    return d

def start_eden_2d_in_3d():
    """eden is a dictionary consisted of such items: (x, y, z, d): [a, b, c]
    where (x, y, z) are the coordinates of the square center
    d is indicator which plane the square is parallel to
    a = 1 if the square is already in the complex (0 if only in the perimeter)
    b = number of neighbours already in the complex (from 0 to 12)
    c = 1 if tile is in inner perimeter """
    """perimeter is a layer of the squares lying on the perimeter (but not yet it the complex)"""
    eden = {(0, 0, 0.5, 2): [1, 0, 0],
            (1, 0, 0.5, 2): [0, 1, 0],
            (-1, 0, 0.5, 2): [0, 1, 0],
            (0, 1, 0.5, 2): [0, 1, 0],
            (0, -1, 0.5, 2): [0, 1, 0],
            (0.5, 0, 1, 0): [0, 1, 0],
            (0, 0.5, 1, 1): [0, 1, 0],
            (-0.5, 0, 1, 0): [0, 1, 0],
            (0, -0.5, 1, 1): [0, 1, 0],
            (0.5, 0, 0, 0): [0, 1, 0],
            (0, 0.5, 0, 1): [0, 1, 0],
            (-0.5, 0, 0, 0): [0, 1, 0],
            (0, -0.5, 0, 1): [0, 1, 0]}
    perimeter = list(eden.keys())
    perimeter.remove((0, 0, 0.5, 2))
    return eden, perimeter

def shift_for_neighbors(third_direction):
    directions = [0, 1, 2]
    directions.remove(third_direction)

    neighbours_parallel = np.array([[0, 0, 0, third_direction]] * 4)
    diff_parallel = np.array([[0, 1.], [1, 0], [0, -1], [-1, 0]]).astype(float)
    for i, n in enumerate(neighbours_parallel):
        n[directions] = diff_parallel[i]

    diff_perp = [0, 0, 0, 0]
    diff_perp[int(third_direction)] = 0.5
    neighbours_plus = neighbours_parallel * 0.5 + diff_perp
    neighbours_minus = neighbours_parallel * 0.5 - diff_perp
    neighbours_perp = np.concatenate((neighbours_plus, neighbours_minus))

    diff_nearest_tiles = np.concatenate((neighbours_parallel, neighbours_perp))
    return diff_nearest_tiles

def actualize_neighbors(tile_selected, eden, perimeter, shift_neighbors):
    tile = list(tile_selected)
    eden[tile_selected][2] = 0
    directions = [0, 1, 2]
    directions.remove(tile[3])

    diff_nearest_tiles = shift_neighbors[int(tile[3])]
    nearest_tiles = diff_nearest_tiles + (tile[:3]+[0])
    for n in nearest_tiles:
        n[3] = dimension(n[:3])
    nearest_tiles = [tuple(n) for n in nearest_tiles]
    nearest_n = [0]*4
    for i, n in enumerate(nearest_tiles):
        # if n[2] <= 0:
        #     continue
        if n in eden:
            # if n in perimeter:
            #     holes_voids = [v for v in voids if voids[v][2] != 0]
            #     v = nearest_voids(n)
            #     if v[0] in holes_voids and v[1] in holes_voids:
            #         z = 1
            #     else:
            #         z = 0
            #     eden[n][2] = z
            eden[n][1] += 1
            if eden[n][0] == 1:
                if diff_nearest_tiles[i][directions[1]] > 0:
                    nearest_n[0] = 1
                if diff_nearest_tiles[i][directions[0]] > 0:
                    nearest_n[1] = 1
                if diff_nearest_tiles[i][directions[1]] < 0:
                    nearest_n[2] = 1
                if diff_nearest_tiles[i][directions[0]] < 0:
                    nearest_n[3] = 1
        else:
            eden[n] = [0, 1, 0]
            perimeter = perimeter + [n]
            # holes_voids = [v for v in voids if voids[v][2] != 0]
            # v = nearest_voids(n)
            # if v[0] in holes_voids:
            #     z = 1
            # else:
            #     z = 0
            # eden[n] = [0, 1, z]
    return eden, perimeter, nearest_n, nearest_tiles

def shift_for_neighbours_diag(third_direction):
    directions = [0, 1, 2]
    directions.remove(third_direction)

    diff_diag_parallel = np.array([[0, 0, 0, third_direction]] * 4)
    diff_parallel = np.array([[1., 1.], [1, -1.], [-1., -1], [-1, 1.]]).astype(float)
    for i, n in enumerate(diff_diag_parallel):
        n[directions] = diff_parallel[i]

    diff_diag_perp = np.array([[0.5, 1.],
                               [1., 0.5],
                               [1., -0.5],
                               [0.5, -1],
                               [-0.5, -1.],
                               [-1., -0.5],
                               [-1., 0.5],
                               [-0.5, 1.],
                               ]).astype(float)
    diff_diag_plus = np.array([[0., 0., 0., 0.]]*8)
    diff_diag_minus = np.array([[0., 0., 0., 0.]]*8)

    for i in range(8):
        diff_diag_plus[i][directions] = diff_diag_perp[i]
        diff_diag_plus[i][int(third_direction)] = 0.5
        diff_diag_minus[i][directions] = diff_diag_perp[i]
        diff_diag_minus[i][int(third_direction)] = -0.5

    diff_diag_perp = np.concatenate((diff_diag_plus, diff_diag_minus))

    diff_diag_all = np.concatenate((diff_diag_parallel, diff_diag_perp))
    return diff_diag_all

def neighbours_diag(tile_selected, eden, shift_diag_neighbours):
    """For the Euler Characteristic"""
    tile = list(tile_selected)
    directions = [0, 1, 2]
    directions.remove(tile[3])

    diff_diag_all = shift_diag_neighbours[int(tile[3])]
    nearest_diag_tiles = diff_diag_all + (tile[:3]+[0])
    for n in nearest_diag_tiles:
        n[3] = dimension(n[:3])
    nearest_diag_tiles = list(map(tuple, nearest_diag_tiles))

    nearest_diag = [0] * 4
    for i in range(0, len(nearest_diag_tiles)):
        if nearest_diag_tiles[i] in eden:
            if eden[nearest_diag_tiles[i]][0] == 1:
                if np.array_equal((diff_diag_all[i][directions] > 0), [True, True]):
                    nearest_diag[0] = 1
                elif np.array_equal((diff_diag_all[i][directions] > 0), [True, False]):
                    nearest_diag[1] = 1
                elif np.array_equal((diff_diag_all[i][directions] > 0), [False, False]):
                    nearest_diag[2] = 1
                else:
                    nearest_diag[3] = 1
    return nearest_diag, nearest_diag_tiles

def actualize_vef(vertices, edges, nearest_n, nearest_diag):
    """for Euler characteristics. function updates number of vertices and edges"""
    v = [1] * 4
    e = [1] * 4
    for i in range(4):
        if nearest_diag[i] == 1:
            v[i] = 0
        if nearest_n[i] == 1:
            e[i] = 0
            v[i-1] = 0
            v[i] = 0
    vertices = vertices + sum(v)
    edges = edges + sum(e)

    return vertices, edges

def nearest_voids(tile):
    diff = np.array([[0, 0, 0]] * 2).astype(float)
    diff[:, int(tile[3])] = [0.5, -0.5]
    voids = [tuple(tile[:3] + diff[0]), tuple(tile[:3] + diff[1])]
    return voids

def nearest_cubes(tile):
    tile = np.array(tile).astype(float)
    directions = [0, 1, 2]
    directions.remove(tile[3])
    third_direction = int(tile[3])

    # upper cube
    cube_1 = np.array([[0, 0, 0, 0]]*6).astype(float)
    cube_1[0][directions] = tile[directions]
    cube_1[0][third_direction] = tile[third_direction] + 1

    diff = np.array([[0, 0.5], [0.5, 0], [0, -0.5], [-0.5, 0]]).astype(float)
    for i, n in enumerate(cube_1):
        if 0 < i < 5:
            n[directions] = tile[directions] + diff[i-1]
            n[third_direction] = tile[third_direction] + 0.5
    cube_1[5] = tile
    for n in cube_1:
        n[3] = dimension(n[:3])
    cube_1 = [tuple(n) for n in cube_1]

    # lower cube
    cube_2 = np.array([[0, 0, 0, 0]]*6).astype(float)
    cube_2[0][directions] = tile[directions]
    cube_2[0][third_direction] = tile[third_direction] - 1

    diff = np.array([[0, 0.5], [0.5, 0], [0, -0.5], [-0.5, 0]]).astype(float)
    for i, n in enumerate(cube_2):
        if 0 < i < 5:
            n[directions] = tile[directions] + diff[i-1]
            n[third_direction] = tile[third_direction] - 0.5
    cube_2[5] = tuple(tile)
    for n in cube_2:
        n[3] = dimension(n[:3])
    cube_2 = [tuple(n) for n in cube_2]
    return cube_1, cube_2

def check_cube_in_eden(cube, eden):
    c = True
    for n in cube:
        if n not in eden:
            c = False
        else:
            if eden[n][0] == 0:
                c = False
    return c

def update_void_dict(v, c, eden):
    faces = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    for j, void in enumerate(v):
        for face in c[j]:
            if face in eden:
                if eden[face][0] == 1:
                    for i in range(3):
                        if face[3] == i:
                            if face[i] < void[i]:
                                faces[j][2*i] = 1
                            else:
                                faces[j][2*i+1] = 1
                            if face[i] == void[i]:
                                sys.exit('Something wrong with update_void_dict')
    return faces

def shift_vertices(third_direction):
    directions = [0, 1, 2]
    directions.remove(third_direction)
    diff = np.array([[1., 1.], [1., -1.], [-1., -1.], [-1., 1.]]) * 0.45
    shift = np.array([[0.]*3]*4)
    for i in range(4):
        shift[i][directions] = diff[i]
    return shift

def return_vertices(tile, shift):
    center = np.array(tile[:3])
    third_direction = tile[3]
    shift = shift[int(third_direction)]
    vertices = [tuple(x) for x in center+shift]
    return vertices

def euler_characteristic(k0, k1, k2, k3):
    return k0 - k1 + k2 - k3

def return_betti_1(betti_2, euler_ch):
    return 1 + betti_2 - euler_ch

def shift_for_edge_voids(third_direction):
    directions = [0, 1, 2]
    directions.remove(third_direction)
    diff = np.array([[0, 1.], [1, 0], [0, -1], [-1, 0]]).astype(float)

    cubes_plus = np.array([[0, 0, 0]] * 4).astype(float)
    for i, n in enumerate(cubes_plus):
        n[directions] = diff[i]
        n[third_direction] = 0.5
    cubes_minus = np.array([[0, 0, 0]] * 4).astype(float)
    for i, n in enumerate(cubes_minus):
        n[directions] = diff[i]
        n[third_direction] = -0.5
    return np.concatenate((cubes_plus, cubes_minus))

def edge_voids(tile, shift_):
    """returns 10 voids which share at least an edge with the tile"""
    tile = np.array(tile).astype(float)
    cubes = []
    """first two cubes which share the tile"""
    cubes[:1] = nearest_voids(tile)
    """other 8 cubes which share the edge"""
    shift = shift_[int(tile[3])]
    cubes[2:10] = tile[:3] + shift
    cubes = [tuple(x) for x in cubes]
    return cubes

def barcode_forest(barcode, tags):
    bars_pure = []
    bars_hole = []
    for x in barcode:
        if barcode[x][0] == 0:
            bars_pure += [tuple([2, tuple(barcode[x][1])])]

    for x in tags:
        b = {}
        for elem in barcode:
            if barcode[elem][2][0] == x:
                b[tuple(barcode[elem][2])] = barcode[elem][1]
        bars_hole += bars_from_tree(b, x)
    return bars_pure + bars_hole

def hamming2(s1, s2):
    """Calculates the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def bars_from_tree(b, tag):
    n = max(len(x) for x in b)
    bars = []
    while n > 0:
        leaves_parent = [x for x in b if len(x) == n - 1]

        # print(leaves_parent)
        possible_leaves = [x for x in b if len(x) == n]
        # print(possible_leaves)
        for j in leaves_parent:
            leaves = []
            for x in possible_leaves:
                # print(x)
                root = list(x)
                del root[-1]
                root = tuple(root)
                if hamming2(j, root) == 0:
                    leaves = leaves + [x]
            # diff(possible_leaves, leaves)
            # print(leaves)
            if len(leaves) > 0:
                times = []
                for x in leaves:
                    times = times + [b[x][1]]
                if 0 in times:
                    ind = times.index(0)
                    for i in range(0, len(leaves)):
                        if i != ind:
                            bars += [tuple([2, tuple(b[leaves[i]])])]
                else:
                    ind = times.index(max(times))
                    for i in range(0, len(leaves)):
                        if i != ind:
                            bars = bars + [tuple([2, tuple(b[leaves[i]])])]
                    b[j][1] = max(times)
        n = n - 1
    bars += [tuple([2 ,tuple(b[(tag,)])])]
    return bars

def num_holes(created_holes, holes):
    tricube = dict.fromkeys(['l','i'], 0)
    tricube_f = dict.fromkeys(['l','i'], 0)
    tetracube = dict.fromkeys(['I', 'L', 'T', 'O', 'Z', 'A1', 'A2'], 0)
    tetracube_f = dict.fromkeys(['I', 'L', 'T', 'O', 'Z', 'A1', 'A2'], 0)

    total_3 = [i for i in created_holes if i[-2] == 3]
    final_3 = [holes[i] for i in holes if len(holes[i]) == 3]
    total_4 = [i for i in created_holes if i[-2] == 4]
    final_4 = [holes[i] for i in holes if len(holes[i]) == 4]

    # tricube case
    for x in total_3:
        j = 0
        for i in range(3):
            if x[-3][0][i] == x[-3][1][i] == x[-3][2][i]:
                j += 1
                dim = i
        long = j == 2
        tricube['i'] += long
        tricube['l'] += (1 - long)
    for x in final_3:
        j = 0
        for i in range(3):
            if x[0][i] == x[1][i] == x[2][i]:
                j += 1
                dim = i
        long = j == 2
        tricube_f['i'] += long
        tricube_f['l'] += (1 - long)
    # tetracube case
    for x in total_4:
        dist = distances(x[-3])
        if dist == [2,2,3]:
            typ = 'I'
        elif dist == [1.4, 2, 2.2]:
            typ = 'L'
        elif dist == [1.4, 1.4, 2]:
            typ = 'T'
        elif dist == [1, 1.4, 1.4]:
            typ = 'O'
        elif dist == [1.4, 1.4, 2.2]:
            typ = 'Z'
        elif dist == [1.4, 1.4, 1.4]:
            typ = 'A2'
        else:
            typ = 'A1'
        tetracube[typ] += 1
    for x in final_4:
        dist = distances(x)
        if dist == [2,2,3]:
            typ = 'I'
        elif dist == [1.4, 2, 2.2]:
            typ = 'L'
        elif dist == [1.4, 1.4, 2]:
            typ = 'T'
        elif dist == [1, 1.4, 1.4]:
            typ = 'O'
        elif dist == [1.4, 1.4, 2.2]:
            typ = 'Z'
        elif dist == [1.4, 1.4, 1.4]:
            typ = 'A2'
        else:
            typ = 'A1'
        tetracube_f[typ] += 1
    return tricube, tricube_f, tetracube, tetracube_f

def distances(hole):
    hole = [np.array(k) for k in hole]
    dist = []
    for i in range(4):
        for j in range(4):
            if i < j:
                dist.append(np.linalg.norm(hole[i]-hole[j]))
    dist.sort()
    dist = [round(i, 1) for i in dist]
    return dist[3:]

def tiles_from_voids(voids):
    tiles = set()
    for x in voids:
        x = np.array(x)
        x = np.concatenate((x, [0]))
        shift = [[0.5, 0, 0, 0],
                 [-0.5, 0, 0, 0],
                 [0, 0.5, 0, 0],
                 [0, -0.5, 0, 0],
                 [0, 0, -0.5, 0],
                 [0, 0, -0.5, 0]]
        shell = x + shift
        for n in shell:
            n[3] = dimension(n[:3])
        shell = set([tuple(x) for x in shell])
        tiles = tiles.union(shell)
    return list(tiles)

def final_inner_2d(holes, perimeter, eden):
    inn = []
    holes2 = [holes[x] for x in holes if len(holes[x]) > 1]
    for hole in holes2:
        tiles = tiles_from_voids(hole)
        for x in tiles:
            if x in perimeter:
                inn.append(x)
    return inn


"""DRAWING and PLOTTING"""
def draw_square(x0, y0, z0, d, ax, alpha=1, col='gray', ls=0.47):
    """With center at x, y, z draw a square of area ls^2"""
    """d = 1 if square is parallel to xOy, d = 2 if x0z, d = 3 if y0z"""
    """ls is a half of square side"""
    if d == 0:
        # col = 'blue'
        y = np.linspace(y0-ls, y0+ls, num=2)
        z = y + z0 - y0
        y, z = np.meshgrid(y, z)
        x = np.ones((y.shape[0], y.shape[1])) * x0
        ax.plot_surface(x, y, z, color=col, alpha=alpha, linewidth=0, antialiased=True)
    if d == 1:
        # col = 'red'
        x = np.linspace(x0-ls, x0+ls, num=2)
        z = x + z0 - x0
        x, z = np.meshgrid(x, z)
        y = np.ones((x.shape[0], x.shape[1])) * y0
        ax.plot_surface(x, y, z, color=col, alpha=alpha, linewidth=0, antialiased=True)
    if d == 2:
        x = np.linspace(x0-ls, x0+ls, num=2)
        y = x + y0 - x0
        x, y = np.meshgrid(x, y)
        z = np.ones((x.shape[0], x.shape[1])) * z0
        ax.plot_surface(x, y, z, color=col, alpha=alpha, linewidth=0, antialiased=True)

def add_box(eden, ax, max_range=5):
    # Create cubic bounding box to simulate equal aspect ratio
    points = np.array([x for x in eden if eden[x][0] == 1])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    max_range = np.array([x.max()-x.min()+1, y.max()-y.min()+1, z.max()-z.min()+1]).max()/2

    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def draw_eden(eden, folder_name):
    # ax.grid(True)
    # plt.style.use('ggplot')
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.axis('off')
    fig.set_facecolor('w')
    ax.set_facecolor('w')
    ax.grid(False)
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    add_box(eden, ax)

    for x in eden:
        if eden[x][0] == 1:
            draw_square(x[0], x[1], x[2], x[3], ax=ax, col='grey')#'tab:blue')
    draw_square(0, 0, 0.5, 2, ax=ax, col='green')
    plt.show()
    plt.savefig(folder_name+'/eden.png', format='png', dpi=1200)
    plt.close()

def draw_complex(eden, time, tile=None):
    # ax.grid(True)
    plt.style.use('ggplot')
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.axis('off')
    fig.set_facecolor('w')
    ax.set_facecolor('w')
    ax.grid(False)
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    # add_box(eden, ax)
    #
    # add_box(eden, ax, 5)

    for x in eden:
        # if eden[x][0] == 1 and x != tile:
        draw_square(x[0], x[1], x[2], x[3], ax=ax, alpha=0.3, col='dimgray')
        # if eden[x][0] == 0:
        #     draw_square(x[0], x[1], x[2], x[3], ax=ax, alpha=0.3, col='lightgrey')
    for x in tile:
        draw_square(x[0], x[1], x[2], x[3], ax=ax, alpha=1, col='dimgrey')
    # draw_square(0, 0, 0.5, 2, ax=ax, col='green')
    # draw_square(tile[0], tile[1], tile[2], tile[3], ax=ax, col='darkorange')
    plt.savefig('pictures/eden_' + str(time) + '.png', format='png', dpi=1200)

def draw_barcode(barcode, time, folder_name):
    if not barcode:
        print("Model is too small. There is no barcode.")
        return
    fig = plt.figure()
    plt.style.use('ggplot')
    # plt.axis('off')
    plt.grid(True)
    plt.rc('grid', linestyle="-", color='gray')
    plt.yticks([])
    plt.gca().set_aspect('equal', adjustable='box')
    i = 0
    for x in barcode:
        if x[1] == float('inf'):
            plt.plot([x[0], time], [i, i], 'k-', lw=2)
        else:
            plt.plot([x[0], x[1]], [i, i], 'k-', lw=2)
        i = i + 40
    fig.suptitle(r'Persistence Barcode $\beta_2$')
    fig.savefig(folder_name+'/barcode.png', format='png', dpi=1200)
    plt.rcParams.update(plt.rcParamsDefault)
    plt.close()

def draw_barcode_gudhi(barcode, folder_name):
    fig, ax = plt.subplots()
    gd.plot_persistence_barcode(persistence=barcode, max_barcodes=1000)
    ax.set_title(r'Persistence Barcode $\beta_2$')
    plt.savefig(folder_name+'/barcode_2.png', dpi=1200)
    plt.close()

def draw_frequencies_1(dict, folder_name):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    l = len(dict[0])

    sh = []
    for j in np.arange(-1, 3):
        sh.append(next((i for i, x in enumerate(dict[j]) if x), 0))
    shift = max(sh)

    ax.plot(range(shift, l), dict[-1][shift:], color='tab:red', label='-1',  linewidth=0.75)
    ax.plot(range(shift, l), dict[0][shift:], color='tab:orange', label='0',  linewidth=0.75)
    ax.plot(range(shift, l), dict[1][shift:], color='tab:green', label='+1',  linewidth=0.75)
    ax.plot(range(shift, l), dict[2][shift:], color='tab:blue', label='+2',  linewidth=0.75)

    plt.yscale('log')
    ax.set_ylabel(r'Frequency of Change in $\beta_1$')
    ax.set_xlabel('t')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.legend(loc=1, prop={'size': 6})
    fig.savefig(folder_name+'/fr_b_1.png', format='png', dpi=1200)
    plt.close()

def draw_frequencies_2(dict, changes, folder_name):
    fig, ax = plt.subplots()
    l = len(dict[0])

    ch_1 = [i for i, j in enumerate(changes) if j == -1]
    y_1 = []
    for x in ch_1:
        y_1 += [dict[-1][x+1]]

    sh = []
    for j in np.arange(-1, 2):
        sh.append(next((i for i, x in enumerate(dict[j]) if x), 0))
    shift = max(sh)

    if next((i for i, x in enumerate(dict[-1]) if x), 0) != 0:
        ax.plot(range(shift, l), dict[-1][shift:], color='tab:purple', label='-1', linewidth=0.75)
    ax.plot(range(shift, l), dict[0][shift:], color='tab:orange', label='0', linewidth=0.75)
    ax.plot(range(shift, l), dict[1][shift:], color='tab:green', label='+1', linewidth=0.75)

    plt.yscale('log')
    ax.set_ylabel(r'Frequency of Change in $\beta_2$')
    ax.set_xlabel('t')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.legend(loc=1, prop={'size': 6})
    fig.savefig(folder_name+'/fr_b_2.png', format='png', dpi=1200)
    plt.close()

def draw_frequencies_2_eu(dict, changes, folder_name):
    fig, ax = plt.subplots()

    ch_1 = [i for i, j in enumerate(changes) if j == 1]
    y_1 = []
    for x in ch_1:
        y_1 += [dict[1][x+1]]

    if next((i for i, x in enumerate(dict[1]) if x), 0) != 0:
        ax.scatter(ch_1, y_1, s=5, marker='o', color="tab:purple", label='+1')

    plt.yscale('log')
    ax.set_ylabel(r'Frequency of Change in $\beta_2$')
    ax.set_xlabel('t')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.legend(loc=1, prop={'size': 6})
    fig.savefig(folder_name+'/fr_b_2.png', format='png', dpi=1200)
    plt.close()

def draw_frequencies_1_eu(dict, changes, folder_name):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    l = len(dict[0])

    ch_1 = [i for i, j in enumerate(changes) if j == -1]
    y_1 = []
    for x in ch_1:
        y_1 += [dict[1][x+1]]

    sh = []
    for j in np.arange(0, 3):
        sh.append(next((i for i, x in enumerate(dict[j]) if x), 0))
    shift = max(sh)

    ax.plot(range(shift, l), dict[0][shift:], color='tab:orange', label='0',  linewidth=0.75)
    ax.plot(range(shift, l), dict[1][shift:], color='tab:green', label='+1',  linewidth=0.75)
    ax.plot(range(shift, l), dict[2][shift:], color='tab:blue', label='+2',  linewidth=0.75)

    if next((i for i, x in enumerate(dict[-1]) if x), 0) != 0:
        plt.scatter(ch_1, y_1, s=5, marker='o', color="tab:red", label='-1')

    plt.yscale('log')
    ax.set_ylabel(r'Frequency of Change in $\beta_1$')
    ax.set_xlabel('t')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.legend(loc=1, prop={'size': 6})
    fig.savefig(folder_name+'/fr_b_1.png', format='png', dpi=1200)
    plt.close()

def draw_diagram_holes(created_holes, holes, folder_name):
    fr_cr = [created_holes[i][-2] for i in range(len(created_holes))]
    fr_cr.sort()
    fr_final = [len(holes[i]) for i in holes]
    fr_final.sort()

    counter_cr = collections.Counter(fr_cr)
    for j in range(1, list(counter_cr.keys())[-1]):
        if j not in counter_cr:
            counter_cr[j] = 0

    counter_final = collections.Counter(fr_final)
    for i in counter_cr.keys():
        if i not in counter_final:
            counter_final[i] = 0
    width = 0.35

    labels = range(len(counter_cr.keys())+1)
    x = np.arange(len(labels))

    fig, ax = plt.subplots()
    plt.yscale('log')
    ax.bar(np.array(list(counter_cr.keys())) - width/2, counter_cr.values(), width, color=[(0.44, 0.57, 0.79)], label='Total')
    ax.bar(np.array(list(counter_final.keys())) + width/2, counter_final.values(), width, color=[(225/256, 151/256, 76/256)], label='Final')

    ax.set_ylabel('Frequency of Number of Holes')
    ax.set_xlabel('Volume')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if len(labels) >= 70:
        plt.setp(ax.get_xticklabels(), fontsize=2)
    elif len(labels) >= 60:
        plt.setp(ax.get_xticklabels(), fontsize=3)
    elif len(labels) >= 50:
        plt.setp(ax.get_xticklabels(), fontsize=4)
    elif len(labels) >= 45:
        plt.setp(ax.get_xticklabels(), fontsize=5)
    elif len(labels) >= 40:
        plt.setp(ax.get_xticklabels(), fontsize=6)
    elif len(labels) >= 30:
        plt.setp(ax.get_xticklabels(), fontsize=7)
    elif len(labels) >= 20:
        plt.setp(ax.get_xticklabels(), fontsize=6)

    ax.legend(loc=1)
    fig.tight_layout()
    fig.savefig(folder_name+'/holes.png', format='png', dpi=1200)
    plt.close()

def draw_tri_tetra(tri, tri_f, tetra, tetra_f, folder_name):
    width = 0.35
    labels = list(tri)+list(tetra)
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    plt.yscale('log')

    try:
        ax.bar(x[:2]-width/2, tri.values(), width, label='Tricubes Total', color='navy')
        ax.bar(x[:2]+width/2, tri_f.values(), width, label='Tricubes Final', color='royalblue')
        ax.bar(x[2:]-width/2, tetra.values(), width, label='Tetracubes Total', color='chocolate')
        ax.bar(x[2:]+width/2, tetra_f.values(), width, label='Tetracubes Final', color='orange')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Frequency of Number of Holes')
        ax.set_xlabel('Type of a Hole')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder_name+'/tri-tetra-cubes.png', format='png', dpi=1200)
        plt.close()
    except ValueError:
        print("No tricubes and tetracubes in this complex")
        plt.close()

def plot_b_per(b1, b2, p2, p3, time, N, folder_name):
    n = int(time/10)
    nn = n

    def func(x, a, b, c):
        return a * x ** b + c
    ydata_f = b1
    xdata_f = range(len(ydata_f))
    ydata = ydata_f[N:]
    xdata = xdata_f[N:]
    plt.xscale('log')
    plt.yscale('log')
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.plot(xdata_f[n:], ydata_f[n:], 'm-', label=r'$\beta_1(t)$ data',  linewidth=0.75)
    try:
        popt, pcov = curve_fit(func, xdata, ydata)#, bounds=([0.,0., 1000], [2., 1, 1500]))
    except RuntimeError:
        popt, pcov = curve_fit(func, xdata, ydata, bounds=([0., 0., -10], [10., 10., 900]))

    plt.plot(xdata_f[n:], func(xdata_f[n:], *popt), 'm--', label=r'fit: $y=%5.2f x^{%5.3f}%+5.1f$' % tuple(popt), linewidth=0.75)

    ydata_f = b2
    xdata_f = range(len(ydata_f))
    ydata = ydata_f[N:]
    xdata = xdata_f[N:]
    plt.plot(xdata_f[n:], ydata_f[n:], 'b-', label=r'$\beta_2(t)$ data',  linewidth=0.75)
    try:
        popt, pcov = curve_fit(func, xdata, ydata)
    except RuntimeError:
        popt, pcov = curve_fit(func, xdata, ydata, bounds=([0., 0., -5000], [1., 2, 4000]))
    plt.plot(xdata_f[n:], func(xdata_f[n:], *popt), 'b--', label=r'fit: $y=%5.3f x^{%5.3f}%+5.3f$' % tuple(popt),  linewidth=0.75)

    # Constrain the optimization to the linear function
    try:
        popt, pcov = curve_fit(func, xdata, ydata, bounds=([0., 0., -np.inf], [1., 1., np.inf]))
    except ValueError:
        popt, pcov = curve_fit(func, xdata, ydata, bounds=([0., 0., -5000], [1., 1., 10]))

    plt.plot(xdata_f[nn+n:], func(xdata_f[nn+n:], *popt), 'g--', label=r'fit: $y=%5.3f x^{%5.3f}%+5.3f$' % tuple(popt),  linewidth=0.75)

    def func(x, a, b):
        return a * x ** b
    ydata = p2
    xdata = range(len(ydata))
    plt.plot(xdata[n:], ydata[n:], color='orange', linestyle='solid', label=r'$P_{2}(t)$ data',  linewidth=0.75)
    popt, pcov = curve_fit(func, xdata, ydata)
    plt.plot(xdata[n:], func(xdata[n:], *popt), color='orange', linestyle='dashed', label=r'fit: $y=%5.2f x^{%5.3f}$' % tuple(popt),  linewidth=0.75)
    ydata = p3
    xdata = range(len(ydata))
    plt.plot(xdata[n:], ydata[n:], color='lightblue', linestyle='solid', label=r'$P_{3}(t)$ data',  linewidth=0.75)
    popt, pcov = curve_fit(func, xdata, ydata)
    plt.plot(xdata[n:], func(xdata[n:], *popt), color='lightblue', linestyle='dashed', label=r'fit: $y=%5.2f x^{%5.3f}$' % tuple(popt),  linewidth=0.75)

    plt.xlabel('t')
    plt.ylabel('data')
    plt.legend(loc=4, prop={'size': 6})
    plt.tight_layout()
    plt.savefig(folder_name+'/per-b-time.png', dpi=1200)
    plt.close()

def draw_square_0(x, y, col='gray', alpha=1, ls=0.35):
    """With center at x, y draw a square of area 1"""
    """it's area is actually 4ls^2, isn't it?"""
    """ls is a half of square side"""
    # plt.grid(True)
    plt.fill([x - ls, x + ls, x + ls, x - ls], [y - ls, y - ls, y + ls, y + ls], alpha=alpha, color=col)

