from Drawing import draw


def start_eden_2d_in_3d():
    """eden is a dictionary consisted of such items: (x, y, z, d): [a, b, c]
    where (x, y, z) are the coordinates of the square center
    d is indicator which plane the square is parallel to
    a = 1 if the square is already in the complex (0 if only in the perimeter)
    b = number of neighbours already in the complex (from 0 to 12)
    c = 1 if tile is a part of a hole"""
    """perimeter is a layer of the squares lying on the perimeter (but not yet it the complex)"""
    eden = {(0, 0, 0, 0): [1, 0, 0],
            (1, 0, 0, 0): [0, 1, 0],
            (-1, 0, 0, 0): [0, 1, 0],
            (0, 1, 0, 0): [0, 1, 0],
            (0, -1, 0, 0): [0, 1, 0],
            (0.5, 0, 0.5, 2): [0, 1, 0],
            (0, 0.5, 0.5, 1): [0, 1, 0],
            (-0.5, 0, 0.5, 2): [0, 1, 0],
            (0, -0.5, 0.5, 1): [0, 1, 0],
            (0.5, 0, -0.5, 2): [0, 1, 0],
            (0, 0.5, -0.5, 1): [0, 1, 0],
            (-0.5, 0, -0.5, 2): [0, 1, 0],
            (0, -0.5, -0.5, 1): [0, 1, 0]
            }
    perimeter = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    return eden, perimeter


# Eden = {(2, 0, 0, 0): [1], (0, -3, 1, 1): [1], (-2, 3, 1, 2): [1]}
Eden, p = start_eden_2d_in_3d()
draw(Eden, 0)

