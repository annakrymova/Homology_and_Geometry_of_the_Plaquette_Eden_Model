import numpy as np
import sys


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


def start_eden_2d_in_3d_line(length):
    eden = {(0, 0, 0.5, 2): [1, 0, 0],
            (1, 0, 0.5, 2): [0, 1, 0],
            (-1, 0, 0.5, 2): [0, 1, 0],
            (0, 1, 0.5, 2): [0, 1, 0],
            (0, -1, 0.5, 2): [0, 1, 0],
            (0.5, 0, 1, 0): [0, 1, 0],
            (0, 0.5, 1, 1): [0, 1, 0],
            (-0.5, 0, 1, 0): [0, 1, 0],
            (0, -0.5, 1, 1): [0, 1, 0],
            }
    perimeter = list(eden.keys())
    perimeter.remove((0, 0, 0.5, 2))

    shift_neighbours = [shift_for_neighbors(0), shift_for_neighbors(1), shift_for_neighbors(2)]
    for i in range(length):
        eden[(i+1, 0, 0.5, 2)] = [1, 0, 0]
        eden, perimeter, nearest_n, nearest_neighbour_tiles = actualize_neighbors((i+1, 0, 0.5, 2), eden, perimeter, shift_neighbours)
        eden[(-(i+1), 0, 0.5, 2)] = [1, 0, 0]
        eden, perimeter, nearest_n, nearest_neighbour_tiles = actualize_neighbors((-(i+1), 0, 0.5, 2), eden, perimeter, shift_neighbours)
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


def euler_characteristic(k0, k1, k2):
    return k0 - k1 + k2


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
            bars_pure = bars_pure + [barcode[x][1]]

    for x in tags:
        b = {}
        for elem in barcode:
            if barcode[elem][2][0] == x:
                b[tuple(barcode[elem][2])] = barcode[elem][1]
        bars_hole = bars_hole + bars_from_tree(b, x)
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
        if x[-3] in final_3:
            tricube_f['i'] += long
            tricube_f['l'] += (1 - long)
    # tetracube case
    for x in total_4:
        # check that x is plane
        j = 0
        for i in range(3):
            if x[-3][0][i] == x[-3][1][i] == x[-3][2][i] == x[-3][3][i]:
                j += 1
        if j > -1: # so, the tetracube is plane
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
            if x[-3] in final_4:
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





