def hamming2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def diff(li1, li2):
    """Returns characters that present only in one of two strings"""
    """the function isn't used (the line with it is commented) do we need it?"""
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif


def start_eden_2d():
    eden = {(0, 0): [1, 0, 0],
            (1, 0): [0, 1, 0],
            (-1, 0): [0, 1, 0],
            (0, 1): [0, 1, 0],
            (0, -1): [0, 1, 0]}
    perimeter = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    return eden, perimeter


def actualize_neighbors(tile_selected, eden, perimeter):
        
    n1 = (tile_selected[0] + 1, tile_selected[1])
    n2 = (tile_selected[0] + -1, tile_selected[1])
    n3 = (tile_selected[0], tile_selected[1] + 1)
    n4 = (tile_selected[0], tile_selected[1] - 1)

    nearest_n_tiles = [n1, n2, n3, n4]
    nearest_n = [0, 0, 0, 0]
    for i, n in enumerate(nearest_n_tiles):
        if n in eden:
            eden[n][1] = eden[n][1] + 1
            if eden[n][0] == 1:
                nearest_n[i] = 1
        else:
            eden[n] = [0, 1, eden[tile_selected][2]]
            perimeter = perimeter + [n]   
        
    return eden, perimeter, nearest_n, nearest_n_tiles


def neighbours(eden, tile_selected):
    # For the Euler Characteristic
    # Level one z = z -1, Level two z = 0, Level 3 Z = +1
    """this functions is used in grow_eden_debugging()"""

    l21 = (tile_selected[0] - 1, tile_selected[1] + 1)
    l22 = (tile_selected[0] + 1, tile_selected[1] + 1)
    l23 = (tile_selected[0] - 1, tile_selected[1] - 1)
    l24 = (tile_selected[0] + 1, tile_selected[1] - 1)

    N = [l21, l22, l23, l24]
    n = [0] * 4
    for i in range(0, len(N)):
        if N[i] in eden:
            if eden[N[i]][0] == 1:
                n[i] = 1
    return n


def actualize_vef(vertices, edges, nearest_n, n):
    v = [1] * 4
    e = [1] * 4
    for i in range(4):
        if n[i] == 1:
            v[i] = 0
        if nearest_n[i] == 1:
            e[i] = 0

    vertices = vertices + sum(v)
    edges = edges + sum(e)

    return vertices, edges


def add_neighbours_bds(bds, j, iterations, num_possible_components, merged, finished, eden):  # spreading gas
    tile_selected = bds[j][iterations]

    n1 = (tile_selected[0] + 1, tile_selected[1])
    n2 = (tile_selected[0] + -1, tile_selected[1])
    n3 = (tile_selected[0], tile_selected[1] + 1)
    n4 = (tile_selected[0], tile_selected[1] - 1)

    nearest_n_tiles = [n1, n2, n3, n4]

    nearest_n = [0, 0, 0, 0]

    for i, n in enumerate(nearest_n_tiles):
        if n in eden:
            if eden[n][0] == 0:
                nearest_n[i] = 1
        else:
            nearest_n[i] = 1

    for i in range(0, 4):
        if nearest_n[i] == 1:
            if nearest_n_tiles[i] not in bds[j]:
                bds[j] = bds[j] + [nearest_n_tiles[i]]
            for t in range(0, num_possible_components):
                if nearest_n_tiles[i] in bds[t]:
                    if t < j:
                        merged[j] = 1
                        finished[j] = 1
                    if t > j:
                        merged[t] = 1
                        finished[t] = 1
    return bds, merged, finished


def increment_betti_2(eden, tile_selected, nearest_n, nearest_n_tiles, barcode, time, holes, total_holes, created_holes,
                      tags):
    if eden[tile_selected][2] == 0:
        per = 1  # This is 1 if the tile added was in the out perimeter
    else:
        num_hole = eden[tile_selected][2]
        per = 0
    # In this case the tile added was in a hole
    betti_2 = 0

    if sum(nearest_n) == 4:
        betti_2 = - 1
        barcode[num_hole][1][1] = time + 1  # Are we covering a hole that was never divided?
        holes[num_hole].remove(tile_selected)
        if not holes[num_hole]:
            holes.pop(num_hole)

    if sum(nearest_n) == 3:
        betti_2 = 0
        if per == 0:
            holes[num_hole].remove(tile_selected)
    # print(nearest_n)
    if sum(nearest_n) != 4 and sum(nearest_n) != 3:
        num_possible_components = 0
        bds = []
        iterations = 0
        for i in range(0, 4):
            if nearest_n[i] == 0:
                num_possible_components = num_possible_components + 1
                bds = bds + [[nearest_n_tiles[i]]]

        finished = [0] * num_possible_components
        merged = finished.copy()

        while sum(finished) < num_possible_components - per:
            for j in range(0, num_possible_components):
                if finished[j] == 0:
                    bds, merged, finished = add_neighbours_bds(bds, j, iterations, num_possible_components, merged,
                                                               finished, eden)
                    if (iterations + 1) == len(bds[j]):
                        finished[j] = 1
            iterations = iterations + 1

        betti_2 = (num_possible_components - 1) - sum(merged)
        # print(betti_2, per)
        # At this point we have the bds components and the ones that were not merged will become the holes.
        # Here we actualize the holes and we actualize Hole No in eden.
        if betti_2 == 0:
            if per == 0:
                holes[num_hole].remove(tile_selected)
        else:

            if per == 1:
                for i in range(0, num_possible_components):
                    if finished[i] == 1 and merged[i] == 0:
                        total_holes = total_holes + 1
                        holes[total_holes] = bds[i].copy()

                        for x in bds[i]:
                            if x in eden:
                                eden[x][2] = total_holes
                        barcode[total_holes] = [0, [time + 1, 0], [total_holes]]
                        created_holes = created_holes + [[barcode[total_holes][2], bds[i].copy(), len(bds[i])]]

            else:

                if barcode[num_hole][0] == 0:
                    tags = tags + [num_hole]
                    barcode[num_hole][0] = 1

                holes.pop(num_hole)

                for i in range(0, num_possible_components):
                    if finished[i] == 1 and merged[i] == 0:
                        total_holes = total_holes + 1
                        holes[total_holes] = bds[i].copy()
                        for x in bds[i]:
                            if x in eden:
                                eden[x][2] = total_holes
                        barcode[total_holes] = [1, [time + 1, 0], barcode[num_hole][2] + [total_holes]]
                        created_holes = created_holes + [[barcode[total_holes][2], bds[i].copy(), len(bds[i])]]

    return betti_2, total_holes, eden, barcode, holes, created_holes, tags


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
        # print(b)
        bars_hole = bars_hole + bars_from_tree(b, x)
    return bars_pure + bars_hole


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


def increment_betti_1(vertices, edges, time, betti_2_total):
    return 1 + ((-1) * vertices) + edges + ((-1) * (time + 1))
