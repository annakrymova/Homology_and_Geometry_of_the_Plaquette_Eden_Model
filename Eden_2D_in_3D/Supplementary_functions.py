import numpy as np
import sys
from Drawing import draw_eden


def dimension(coordinates):
    if coordinates[2] % 1 == 0:
        d = 2
    else:
        if coordinates[0] % 1 != 0:
            d = 0
        if coordinates[1] % 1 != 0:
            d = 1
    return d


def start_eden_2d_in_3d():
    """eden is a dictionary consisted of such items: (x, y, z, d): [a, b]
    where (x, y, z) are the coordinates of the square center
    d is indicator which plane the square is parallel to
    a = 1 if the square is already in the complex (0 if only in the perimeter)
    b = number of neighbours already in the complex (from 0 to 12)"""
    """perimeter is a layer of the squares lying on the perimeter (but not yet it the complex)"""
    eden = {(0, 0, 0, 2): [1, 0],
            (1, 0, 0, 2): [0, 1],
            (-1, 0, 0, 2): [0, 1],
            (0, 1, 0, 2): [0, 1],
            (0, -1, 0, 2): [0, 1],
            (0.5, 0, 0.5, 0): [0, 1],
            (0, 0.5, 0.5, 1): [0, 1],
            (-0.5, 0, 0.5, 0): [0, 1],
            (0, -0.5, 0.5, 1): [0, 1],
            (0.5, 0, -0.5, 0): [0, 1],
            (0, 0.5, -0.5, 1): [0, 1],
            (-0.5, 0, -0.5, 0): [0, 1],
            (0, -0.5, -0.5, 1): [0, 1]}
    perimeter = list(eden.keys())
    perimeter.remove((0, 0, 0, 2))
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
    directions = [0, 1, 2]
    directions.remove(tile[3])

    diff_nearest_tiles = shift_neighbors[int(tile[3])]
    nearest_tiles = diff_nearest_tiles + (tile[:3]+[0])
    for n in nearest_tiles:
        n[3] = dimension(n[:3])
    nearest_tiles = [tuple(n) for n in nearest_tiles]
    nearest_n = [0]*4
    for i, n in enumerate(nearest_tiles):
        if n in eden:
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
            eden[n] = [0, 1]
            perimeter = perimeter + [n]
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
    for i, n in enumerate(nearest_diag_tiles):
        n[3] = dimension(n[:3])
    nearest_diag_tiles = [tuple(n) for n in nearest_diag_tiles]

    nearest_diag = [0] * 4
    for i in range(0, len(nearest_diag_tiles)):
        if nearest_diag_tiles[i] in eden:
            if eden[nearest_diag_tiles[i]][0] == 1:
                if np.array_equal((diff_diag_all[i][directions] > 0), [True, True]):
                    nearest_diag[0] = 1
                if np.array_equal((diff_diag_all[i][directions] > 0), [True, False]):
                    nearest_diag[1] = 1
                if np.array_equal((diff_diag_all[i][directions] > 0), [False, False]):
                    nearest_diag[2] = 1
                if np.array_equal((diff_diag_all[i][directions] > 0), [False, True]):
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


def euler_characteristic(k0, k1, k2):
    return k0 - k1 + k2


def return_betti_1(betti_2, euler_ch):
    return 1 + betti_2 - euler_ch
