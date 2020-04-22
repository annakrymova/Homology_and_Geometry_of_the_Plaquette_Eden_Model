def convert_perseus(process):
    min_x = min(process, key=lambda x: x[0])
    max_x = max(process, key=lambda x: x[0])
    min_y = min(process, key=lambda x: x[1])
    max_y = max(process, key=lambda x: x[1])
    dimension = 2
    long = max_x[0] + ((-1) * (min_x[0])) + 1
    wide = max_y[1] + ((-1) * (min_y[1])) + 1
    with open('10000_2_2D_prueba.txt', 'w') as f:
        f.writelines('%d\n' % dimension)
        f.writelines('%d\n' % long)
        f.writelines('%d\n' % wide)
        for i in range(min_y[1], max_y[1] + 1):
            for j in range(min_x[0], max_x[0] + 1):
                # print((i,j))
                if (j, i) in process:
                    f.writelines('%d\n' % process.index((j, i)))
                    # print(process.index((j,i)))
                else:
                    f.writelines('inf\n')
                    # print(-1)


def convert_perseus_2(process):  # todo: add the name of a file as an argument to both functions
    dimension = 2
    with open('10000_11_2D_final.txt', 'w') as f:
        f.writelines('%d\n' % dimension)
        i = 0
        for x in process:
            i = i + 1
            y = (x[0], x[1], i)
            f.writelines('%s %s %s\n' % y)
