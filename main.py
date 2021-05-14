import os
from datetime import datetime
import pandas as pd
import gudhi as gd
import matplotlib.pyplot as plt
import tqdm
import cProfile
import pstats
from line_profiler import LineProfiler
import re
import collections
import numpy as np
from scipy.optimize import curve_fit
import itertools
import csv
from e_2d_in_3d import num_holes, grow_eden, return_frequencies_1, return_frequencies_2, draw_barcode, \
            draw_frequencies_1, draw_frequencies_2, draw_diagram_holes, draw_tri_tetra, plot_b_per, draw_eden, \
            draw_frequencies_1_eu, draw_frequencies_2_eu, draw_barcode_gudhi, create_dist_matrix, plot_per_inner,\
            neighbours_diag, read_barcode_b1_from_file, draw_diagram_holes2, plot_per_inner2, draw_pers_diagram,\
            nearest_voids, get_inner_per, get_inner_per_3, plot_b_per2, draw_frequencies_2_2, plot_b_per3, get_vertices


def read_value(arr):
    while True:
        try:
            x = int(input())
            if x not in arr:
                raise ValueError
            break
        except ValueError:
            print("Oops!  That was no valid number.  Try again...")
    return x


print('Welcome to EDEN 2D in 3D Model!')

print('Do you have a file with a model? \n0 -- you want to generate a new model \n1 -- you have a file')
# file = bool(read_value([0, 1]))
file = bool(0)

print(
    'Which type of model? \n0 -- standard 2d in 3d growth model \n1 -- euler-characteristic mediated 2d in 3d growth model'
    '\n2 -- filled-in cubes 2d in 3d')
# model = bool(read_value([0, 1, 2, 3]))
model = 2

print('Do you want a picture of your model? (with a large model it can take time)  \n0 -- no \n1 -- yes')
# pic = bool(read_value([0, 1]))
pic = bool(1)

if pic:
    print('Do you want Python or MAYA 3D model? (We wouldn\'t recommend Python for large models (more than 500 tiles)).'
          ' \n0 -- Python \n1 -- MAYA')
    # maya = bool(read_value([0, 1]))
    maya = bool(1)

"""NO FILE CASE"""
if not file:
    print('How many tiles would you like in your model?')
    while True:
        try:
            Time = int(input())
            break
        except ValueError:
            print("Oops!  That was no valid number.  Try again...")
    # Time = 20

    # print('How many models would you like to build?')
    # while True:
    #     try:
    #         num_models = int(input())
    #         break
    #     except ValueError:
    #         print("Oops!  That was no valid number.  Try again...")
    num_models = 1
    # cols = ['2d_total', '2d_in', '2d_out', '3d_total', '3d_in', '3d_out']
    # df = pd.DataFrame(columns=cols)
    Per_2d_in_array = np.zeros(shape=(num_models, Time))
    Per_3d_in_array = np.zeros(shape=(num_models, Time))
    # Fr_b1_array = np.zeros(shape=(num_models, Time))
    # Fr_b2_array = np.zeros(shape=(num_models, Time))
    Fr_b1_array = []
    Fr_b2_array = []
    Created_holes_array = []
    Holes_array = []
    Per_2d_array = []
    Per_3d_array = []
    Polycubes = []

    for q in range(num_models):
        print("\nWORKING ON MODEL #" + str(q + 1))
        now = datetime.now()
        dt_string = now.strftime("%d.%m.%Y_%H.%M.%S" + str(q))
        if Time >= 1000:
            t = int(Time / 1000)
            folder_name = str(t) + 'k_' + dt_string
        else:
            t = Time
            folder_name = str(t) + '_' + dt_string

        folder_name_cluster = 'experiments/'+'model'+str(model)+'_'+str(t)+'k/'
        try:
            os.stat(folder_name_cluster)
        except:
            os.mkdir(folder_name_cluster)
        folder_name = folder_name_cluster + folder_name
        os.makedirs(folder_name)



        # print("Building a model...")
        # profile = cProfile.Profile()
        # profile.runcall(grow_eden, Time, model, folder_name)
        # ps = pstats.Stats(profile)
        # ps.sort_stats('calls', 'cumtime')
        # ps.print_stats()

        # lp = LineProfiler()
        # lp.add_function(neighbours_diag)
        # lp_wrapper = lp(grow_eden)
        # lp_wrapper(Time, model, folder_name)
        # lp.print_stats()

        Betti_1_total_vector, Per_2d, Per_3d, Betti_2_total_vector, Eden, Process, Created_holes, Holes, Barcode, \
            Vertices, Process_ripser, Inner_perimeter_2d, Voids, Per_2d_in, Per_3d_in, Inner_perimeter_3d, \
            Per_2d_in_array_, Per_3d_in_array_ , P2_, P3_ = grow_eden(Time, model)

        def create_dist_matrix2(Time, eden, num_vert, folder_name):
            matrix = np.full((num_vert, num_vert), np.inf)
            shift = [np.array([[0, 0.5, 0.5], [0, 0.5, -0.5], [0, -0.5, -0.5], [0, -0.5, 0.5]]),
                     np.array([[0.5, 0, 0.5], [0.5, 0, -0.5], [-0.5, 0, -0.5], [-0.5, 0, 0.5]]),
                     np.array([[0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0], [-0.5, 0.5, 0]])]
            dict_vert = {}
            vertex_num = 0

            print("\nCreating distance matrix")
            # pbar = tqdm(total=Time)

            for t, x in enumerate(eden):
                # pbar.update(1)
                vert = get_vertices(x, shift[int(x[3])])
                for v in vert:
                    if v not in dict_vert:
                        dict_vert[v] = vertex_num
                        vertex_num += 1
                ind = []
                for v in vert:
                    ind.append(dict_vert[v])
                edges = [[ind[0], ind[1]], [ind[1], ind[2]], [ind[2], ind[3]], [ind[3], ind[0]], [ind[0], ind[2]]]
                # for e in edges:
                #     e.sort(reverse=True)
                for e in edges:
                    if e[0] > e[1] and matrix[e[0], e[1]] == float('inf'):
                        matrix[e[0], e[1]] = t+1
                    if e[1] > e[0] and matrix[e[1], e[0]] == float('inf'):
                        matrix[e[1], e[0]] = t+1
            matrix = np.tril(matrix, -1)
            # np.savetxt(folder_name+"/distance_matrix.csv", matrix, delimiter=",")
            np.savetxt(folder_name+'/distance_matrix.txt', matrix, fmt='%1.0f', delimiter=',')
            # for i in range(num_vert):
            # #
            # with open(folder_name+'/distance_matrix.txt', 'wb+') as the_file:
            #     for j, line in enumerate(matrix):
            #         np.savetxt(the_file, line[:j], fmt='%1.0f', delimiter=",", newline='\n')
            # with open(folder_name+'/distance_matrix.txt', 'w+') as f:
            #     for j, line in enumerate(matrix):

            return matrix
        matrix = create_dist_matrix2(Time, Process_ripser, Vertices, folder_name)
        print("dist matrix is created!")
        with open(folder_name+'/barcode1.txt', 'w+') as fp:
            pass

        def draw_barcode_gudhi3(barcode, folder_name, num, fs):
            print("Plotting Betti_2 Barcode...")
            plt.rcParams.update({'font.size': fs, 'font.weight': 'light'})
            fig, ax = plt.subplots()
            # plt.xscale('log')
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            gd.plot_persistence_barcode(persistence=barcode, max_barcodes=100000)
            # plt.xscale('log')
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            if num == 2:
                plt.title(r'Barcode $\beta_2$')
            else:
                plt.title(r'Barcode $\beta_1$')

            plt.rcParams.update({'font.size': fs, 'font.weight': 'light'})
            plt.savefig(folder_name+'/barcode_'+str(num)+'_gudhi.jpg', dpi=100)
            plt.savefig(folder_name+'/barcode_'+str(num)+'_gudhi.pdf', dpi=500)
            plt.close()
        Barcode_finite = [x for x in Barcode if x[1][1] != float('inf')]
        draw_barcode_gudhi3(Barcode_finite,folder_name, 2, 12)

        """BARCODE FOR B1"""
        # folder_name = '.'
        Barcode_b1 = read_barcode_b1_from_file(folder_name)
        Barcode_b1 = [a for a in Barcode_b1 if a[1][1] - a[1][0] != float('inf')]
        draw_barcode_gudhi3(Barcode_b1, folder_name, 1,12)

        def draw_pers_diagram2(barcode1, barcode2, size, folder_name, p2, p3, q, minus, minus2, fs):
            barcode1 = [x[1] for x in barcode1]
            barcode2 = [x[1] for x in barcode2]
            plt.rcParams.update({'font.size': fs, 'font.weight': 'light'})
            # barcode1 = [x for x in barcode1 if x[0] > size / 100 and x[1] - x[0] > size / 1000 and x[1] != float('inf')]
            barcode1 = [x for x in barcode1 if x[0] > size / q and x[1] - x[0] > size / q]
            barcode2 = [x for x in barcode2 if x[0] > size / q and x[1] - x[0] > size / q]
            plt.yscale('log')
            plt.xscale('log')


            x = [x[0] for x in barcode1]
            y = [x[1] - x[0] for x in barcode1]
            plt.scatter(x, y, s=0.2, color='royalblue')
            x0 = [i for i in np.unique(x) if i < size - minus]
            # plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 2))(np.unique(x)), color='navy', label=r'$\beta_1$')
            plt.plot(x0, np.poly1d(np.polyfit(x, y, 2))(x0), color='navy', label=r'$\beta_1$')

            x = [x[0] for x in barcode2]
            y = [x[1] - x[0] for x in barcode2]
            plt.scatter(x, y, s=0.2, color='limegreen')
            x0 = [i for i in np.unique(x) if i < size - minus2]
            plt.plot(x0, np.poly1d(np.polyfit(x, y, 2))(x0), color='lime', label=r'$\beta_2$')
            # plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 2))(np.unique(x)), color='forestgreen', label=r'$\beta_2$')

            n = int(size/q)
            x = list(range(size))[n:]
            y = p2[n:]

            plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 2))(np.unique(x)), color='red', label=r'$P_2$')

            # n = int(size/q)
            # x = list(range(size))[n:]
            # y = p3[n:]
            # plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 2))(np.unique(x)), color='red', label=r'$P_2$')

            plt.xlabel('Birth')
            plt.ylabel('Death - Birth')
            plt.title('Persistence Diagram')
            plt.legend(loc='lower left')
            plt.tight_layout()
            plt.savefig(folder_name+'/pers-d.png', dpi=200)
            plt.savefig(folder_name+'/pers-d.pdf', dpi=500)
            plt.close()
        draw_pers_diagram2(Barcode_b1, Barcode_finite, Time, folder_name, Per_2d, Per_3d, 500, 3870, 7000, 12)
        a=10


        # with open(folder_name_cluster+'/per_inner2_array.csv', 'a+') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(Per_2d_in_array_)
        # with open(folder_name_cluster+'/per_inner3_array.csv', 'a+') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(Per_3d_in_array_)
        # with open(folder_name_cluster+'/per2_array.csv', 'a+') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(P2_)
        # with open(folder_name_cluster+'/per3_array.csv', 'a+') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(P3_)
        #
        # Per_2d_in_array[q] = Per_2d_in
        # Per_3d_in_array[q] = Per_3d_in
        #
        # Per_2d_array += [Per_2d[-1]]
        # Per_3d_array += [Per_3d[-1]]
        #
        # fr_cr = [Created_holes[i][-2] for i in range(len(Created_holes))]
        # fr_cr.sort()
        # counter_cr = collections.Counter(fr_cr)
        #
        # fr_f = [len(Holes[i]) for i in Holes]
        # fr_f.sort()
        # counter_f = collections.Counter(fr_f)
        #
        # Created_holes_array += [counter_cr]
        # Holes_array += [counter_f]

        # if model != 1:
        #     def plot_per_inner2(p2, p3, time, folder_name):
        #         def func(x, a, b):
        #             return a * x ** b
        #
        #         plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        #
        #         linew = 1
        #
        #         ydata = p3
        #         xdata = range(len(ydata))
        #         plt.plot(xdata, ydata, color='forestgreen', linestyle='solid', label=r'inner $P_{3}(t)$ data',  linewidth=linew)
        #         # popt, pcov = curve_fit(func, xdata, ydata)
        #         # plt.plot(xdata, func(xdata, *popt), color='lightgreen', linestyle='dashed', label=r'fit: $y=%5.4f x^{%5.3f}$' % tuple(popt),  linewidth=linew)
        #
        #         ydata = p2
        #         xdata = range(len(ydata))
        #         plt.plot(xdata, ydata, color='mediumorchid', linestyle='solid', label=r'inner $P_{2}(t)$ data',  linewidth=linew)
        #         # popt, pcov = curve_fit(func, xdata, ydata)
        #         # plt.plot(xdata, func(xdata, *popt), color='mediumorchid', linestyle='dashed', label=r'fit: $y=%5.6f x^{%5.3f}$' % tuple(popt),  linewidth=linew)
        #         part = 30
        #         mean_p2 = sum(p2[-int(len(p2)/part):])/len(p2[-int(len(p2)/part):])
        #         mean_p3 = sum(p3[-int(len(p3)/part):])/len(p3[-int(len(p3)/part):])
        #
        #         plt.plot(range(time), [mean_p3]*time, color='forestgreen', linestyle='--', linewidth=0.75)
        #         plt.plot(range(time), [mean_p2]*time, color='mediumorchid', linestyle='--', linewidth=0.75)
        #
        #         plt.xlabel('t')
        #         plt.ylabel('Fraction of the Perimeter')
        #         plt.legend(loc=4, prop={'size': 6})
        #         # plt.tight_layout()
        #         my_yticks = [mean_p2, mean_p3]
        #         my_yticks2 = [round(x, 3) for x in my_yticks]
        #         plt.yticks(my_yticks, my_yticks2)
        #         plt.savefig(folder_name+'/per-inner.png', dpi=500)
        #         plt.savefig(folder_name+'/per-inner.pdf', dpi=500)
        #         plt.close()
        #     plot_per_inner2(Per_2d_in, Per_3d_in, Time, folder_name)

        """DF b1 b2 per2 per3"""
        cols = ['b1', 'b2', 'p2', 'p3']
        df_all = pd.DataFrame(columns=cols)
        df_all['b1'] = Betti_1_total_vector
        df_all['b2'] = Betti_2_total_vector
        df_all['p2'] = Per_2d
        df_all['p3'] = Per_3d
        df_all.to_csv(folder_name+r'/b_p_' + str(Time) + '.csv', mode='a+', header=True)
        with open(folder_name_cluster+'/b1.csv', 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(Betti_1_total_vector)
        with open(folder_name_cluster+'/b2.csv', 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(Betti_2_total_vector)
        with open(folder_name_cluster+'/p2.csv', 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(Per_2d)
        with open(folder_name_cluster+'/p3.csv', 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(Per_3d)

        # """read b1"""
        # df = pd.read_csv('df_freq_b1_500000.csv')
        # freq = dict()
        # freq[-1] = df['-1'].tolist()
        # freq[0] = df['0'].tolist()
        # freq[1] = df['1'].tolist()
        # freq[2] = df['2'].tolist()
        # draw_frequencies_1(freq, folder_name)
        # """read b2"""
        # df = pd.read_csv('df_freq_b2_500000.csv')
        # freq = dict()
        # freq[-1] = df['-1'].tolist()
        # freq[0] = df['0'].tolist()
        # freq[1] = df['1'].tolist()
        # freq[2] = df['2'].tolist()
        # draw_frequencies_1(freq, folder_name)
        # Betti_2_total_vector = df['b2'].tolist()
        # p2d = df['p2'].tolist()
        # p3d = df['p3'].tolist()
        # df = df.append(dict(zip(cols, per)), ignore_index=True)

        # draw_eden(Eden, folder_name, Time)
        #
        # """PERIMETER STATISTICS"""
        # per = [Per_2d[-1], len(Inner_perimeter) / Per_2d[-1], (Per_2d[-1] - len(Inner_perimeter)) / Per_2d[-1],
        #        Per_3d[-1], len([x for x in Voids if Voids[x][2] != 0]) / Per_3d[-1],
        #        len([x for x in Voids if Voids[x][2] == 0]) / Per_3d[-1]]
        #
        # df = df.append(dict(zip(cols, per)), ignore_index=True)

        # create_dist_matrix(Time, Process_ripser, Vertices, folder_name)
        # draw_eden(Eden, folder_name, Time)
        # print(Process)

        """DF freq b1 """
        # freq, changes = return_frequencies_1(Betti_1_total_vector, Time)
        # if model == 1:
        #     draw_frequencies_1_eu(freq, changes, folder_name)
        # else:
        #     draw_frequencies_1(freq, folder_name)
        # cols = list(freq.keys())
        # df = pd.DataFrame(columns=cols)
        # for col in cols:
        #     df[col] = freq[col]
        # df.to_csv(folder_name+r'/df_freq_b1_' + str(Time) + '.csv', mode='w+', header=True)
        # Fr_b1_array += [freq]

        """DF freq b2"""
        # if model != 1:
        #     freq, changes = return_frequencies_2(Betti_2_total_vector, Time)
        #     draw_frequencies_2_2(freq, folder_name)
        #     cols = list(freq.keys())
        #     df = pd.DataFrame(columns=cols)
        #     for col in cols:
        #         df[col] = freq[col]
        #     df.to_csv(folder_name+r'/df_freq_b2_' + str(Time) + '.csv', mode='w+', header=True)
        #     Fr_b2_array += [freq]
        #
        # """SAVE BARCODE TO LIST"""
        # f = open('f2.txt', 'w')
        # for ele in Barcode:
        #     f.write(str(ele)+', ')
        # f.close()
        #
        # def draw_barcode_gudhi(barcode, folder_name, num, fs):
        #     print("Plotting Betti_2 Barcode...")
        #     fig, ax = plt.subplots()
        #     # plt.rcParams.update({'font.size': fs, 'font.weight': 'light'})
        #
        #     gd.plot_persistence_barcode(persistence=barcode, max_barcodes=10000)
        #     # plt.xscale('log')
        #     ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        #     # plt.xscale('log')
        #     if num == 2:
        #         plt.title(r'Persistence Barcode $\beta_2$')
        #     else:
        #         plt.title(r'Persistence Barcode $\beta_1$')
        #     plt.savefig(folder_name+'/barcode_'+str(num)+'_gudhi.png', dpi=500)
        #     plt.close()
        # Barcode = [x for x in Barcode if x[1][1] != float('inf')]
        # draw_barcode_gudhi(Barcode, folder_name, 2, 18)
        def plot_b_per3(b1, b2, p2, p3, time, N0, folder_name, m, fs, lw, n0):
            fig, ax = plt.subplots()
            # n = int(time/10)
            n = n0
            nn = n
            N = 0

            def func2(x, a, b, c):
                return a * x ** b + c

            def func(x, a, b):
                return a * x ** b

            ydata_f = b1
            xdata_f = range(len(ydata_f))
            ydata = ydata_f[N:]
            xdata = xdata_f[N:]
            plt.xscale('log')
            plt.yscale('log')
            # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            plt.plot(xdata_f[n:], ydata_f[n:], 'm-', label=r'$\beta_1(t)$',  linewidth=lw)

            if m != 1:
                try:
                    popt, pcov = curve_fit(func, xdata, ydata)#, bounds=([0.,0., 1000], [2., 1, 1500]))
                except RuntimeError:
                    popt, pcov = curve_fit(func, xdata, ydata, bounds=([0., 0., -10], [10., 10., 900]))

                plt.plot(xdata_f[n:], func(xdata_f[n:], *popt), 'm--', label=r'$\beta_1(t)=%5.2f t^{%5.2f}$' % tuple(popt), linewidth=lw)

                print("plotting b2")
                N = N0
                ydata_f = b2
                xdata_f = range(len(ydata_f))
                ydata = ydata_f[N:]
                xdata = xdata_f[N:]
                plt.plot(xdata_f[n:], ydata_f[n:], 'b-', label=r'$\beta_2(t)$',  linewidth=0.75)
                try:
                    popt, pcov = curve_fit(func2, xdata, ydata)
                except RuntimeError:
                    print("exception")
                    popt, pcov = curve_fit(func2, xdata, ydata, bounds=([0., 0., -5000], [10., 2, 4000]))
                print(popt)
                plt.plot(xdata_f[n:], func2(xdata_f[n:], *popt), 'b--', label=r'$\beta_2(t)=%5.2f t^{%5.2f}%+5.1f$' % tuple(popt),  linewidth=lw)

                # Constrain the optimization to the linear function
                # try:
                #     popt, pcov = curve_fit(func2, xdata, ydata, bounds=([0., 0., -np.inf], [1., 1., np.inf]))
                # except ValueError:
                #     popt, pcov = curve_fit(func2, xdata, ydata, bounds=([0., 0., -5000], [1., 1., 10]))
                #
                # plt.plot(xdata_f[nn+n:], func2(xdata_f[nn+n:], *popt), 'g--', label=r'$y=%5.2f x^{%5.2f}%+5.1f$' % tuple(popt),  linewidth=lw)

            if m == 1:
                try:
                    popt, pcov = curve_fit(func, xdata, ydata)#, bounds=([0.,0., 1000], [2., 1, 1500]))
                except RuntimeError:
                    popt, pcov = curve_fit(func, xdata, ydata, bounds=([0., 0., -10], [10., 10., 900]))
                if popt[1] > 1.05:
                    try:
                        popt, pcov = curve_fit(func2, xdata, ydata)#, bounds=([0.,0., 1000], [2., 1, 1500]))
                    except RuntimeError:
                        popt, pcov = curve_fit(func2, xdata, ydata, bounds=([0., 0., -10], [10., 10., 900]))

                    plt.plot(xdata_f[n:], func2(xdata_f[n:], *popt), 'm--', label=r'$\beta_1(t)=%5.2f t^{%5.2f}%+5.1f$' % tuple(popt), linewidth=lw)
                else:
                    plt.plot(xdata_f[n:], func(xdata_f[n:], *popt), 'm--', label=r'$\beta_1(t)=%5.2f t^{%5.2f}$' % tuple(popt), linewidth=lw)

            print("plotting p2")
            ydata = p2
            xdata = range(len(ydata))
            plt.plot(xdata[n:], ydata[n:], color='orange', linestyle='solid', label=r'$P_{2}(t)$',  linewidth=lw)
            popt, pcov = curve_fit(func, xdata, ydata)
            plt.plot(xdata[n:], func(xdata[n:], *popt), color='orange', linestyle='dashed', label=r'$P_2(t)=%5.2f t^{%5.2f}$' % tuple(popt),  linewidth=lw)
            print("plotting p3")
            ydata = p3
            xdata = range(len(ydata))
            plt.plot(xdata[n:], ydata[n:], color='deepskyblue', linestyle='solid', label=r'$P_{3}(t)$',  linewidth=lw)
            popt, pcov = curve_fit(func, xdata, ydata)
            plt.plot(xdata[n:], func(xdata[n:], *popt), color='deepskyblue', linestyle='dashed', label=r'$P_3(t)=%5.2f t^{%5.2f}$' % tuple(popt),  linewidth=lw)

            # font = {'size': fs}
            # plt.rc('font', **font)

            plt.rcParams.update({'font.size': fs, 'font.weight': 'light'})
            plt.xlabel('t', fontsize=fs)
            plt.ylabel('Growth Rates', fontsize=fs)
            handles, labels = ax.get_legend_handles_labels()
            myorder = [0, 2, 4, 6, 1, 3, 5, 7]
            handles = [handles[i] for i in myorder]
            labels = [labels[i] for i in myorder]
            # plt.legend(handles, labels, prop={'size': 6}, loc='lower right', ncol=2)
            plt.legend(handles, labels, prop={'size': fs-4}, loc='lower right', ncol=2)
            # plt.legend(loc=4, prop={'size': 6})
            # plt.rc('xtick', labelsize=fs)
            # plt.rc('ytick', labelsize=fs)
            ax.tick_params(axis='both', which='major', labelsize=fs)
            ax.tick_params(axis='both', which='minor', labelsize=fs)
            # plt.rc('axes', labelsize=fs)
            # plt.rcParams.update({'font.size': fs, 'font.weight': 'light'})
            plt.tight_layout()
            plt.savefig(folder_name+'/per-b-time.jpg', dpi=100)
            plt.savefig(folder_name+'/per-b-time.pdf', dpi=400)
            print("\n done")
            plt.close()
        print("Plotting the growth rates of Betti numbers and the perimeter...")
        plot_b_per3(Betti_1_total_vector, Betti_2_total_vector, Per_2d, Per_3d, Time, int(Time*0.5), folder_name, model, 18,1, 30000)

        if model != 1:
            print("Plotting the frequency of the number of top dimensional holes for specific shapes with 3 and 4 cells...")
            Tricube, Tricube_f, Tetracube, Tetracube_f = num_holes(Created_holes, Holes)
            Polycubes += [[Tricube, Tricube_f, Tetracube, Tetracube_f]]
            draw_tri_tetra(Tricube, Tricube_f, Tetracube, Tetracube_f, folder_name)
        #     print("Plotting the frequency of the volume of top dimensional \"holes\"...")
        #     draw_diagram_holes(Created_holes, Holes, folder_name)
        #
        # if model != 1:
        #     print("Plotting Betti_2 Barcode...")
        #     if model == 2:
        #         brc = [a[1] for a in Barcode if a[1][1] - a[1][0] != float('inf')]
        #     else:
        #         brc = [a[1] for a in Barcode]
        #     draw_barcode(brc, Time, folder_name)
        #     if model == 2:
        #         Barcode = [a for a in Barcode if a[1][1] - a[1][0] != float('inf')]
        #     draw_barcode_gudhi(Barcode, folder_name, 2)
        #
        # if pic:
        #     if maya:
        #         f = open(folder_name + "/MAYA.txt", "w+")
        #         f.write("import maya.cmds as cmds \n"
        #                 "Eden = " + str(Process) + "\nt = len(Eden)"
        #                                            "\nfor i in range(t):"
        #                                            "\n\tcmds.polyCreateFacet(p = Eden[i])")
        #         f.close()
        #         print("We created txt file \"MAYA\" for you. Just copy paste its content to MAYA!")
        #     else:
        #         draw_eden(Eden, folder_name)
        #         print("Python 3D model is created!")

"""Perimeter and Inner Perimeters N clusters"""
if model != 1:
    if not os.path.exists(folder_name_cluster+'per'):
        os.makedirs(folder_name_cluster+'per')
    plot_per_inner2(np.mean(Per_2d_in_array, axis=0), np.mean(Per_3d_in_array, axis=0), Time, folder_name_cluster+'per')
    df_per_in = pd.DataFrame({'p2_in': [np.mean(Per_2d_in_array, axis=0)[-1], np.std(Per_2d_in_array, axis=0)[-1]],
                              'p3_in': [np.mean(Per_3d_in_array, axis=0)[-1], np.std(Per_3d_in_array, axis=0)[-1]],
                              'p2_total': [np.mean(Per_2d_array), np.std(Per_2d_array)],
                              'p3_total': [np.mean(Per_3d_array), np.std(Per_3d_array)]})
    df_per_in.index = ['Mean', 'SD']
    df_per_in.to_csv(folder_name_cluster+'per'+r'/df_per_inner_' + str(num_models) + '.csv', mode='w+', header=True)

"""PERIMETERS AND BETTI NUMBERS"""
df_b1 = pd.read_csv(folder_name_cluster+'b1.csv', header=None)
df_b2 = pd.read_csv(folder_name_cluster+'b2.csv', header=None)
df_p2 = pd.read_csv(folder_name_cluster+'p2.csv', header=None)
df_p3 = pd.read_csv(folder_name_cluster+'p3.csv', header=None)

plot_b_per(df_b1.mean(), df_b2.mean(), df_p2.mean(), df_p3.mean(), Time, int(Time/3), folder_name_cluster, model)
plot_b_per2(df_b1.mean(), df_b2.mean(), df_p2.mean(), df_p3.mean(), Time, int(Time/3), folder_name, model, 8,1)
a=1
# /Users/annakrymova/TUM/MASTER THESIS/Code/Eden_2D_in_3D/experiments/model1_500k
# /Users/annakrymova/TUM/MASTER THESIS/Code/Eden_2D_in_3D/
# df_b1_ = pd.read_csv('experiments/model2_1000k/b1.csv', header=None)
# df_b2_ = pd.read_csv('experiments/model2_1000k/b2.csv', header=None)
# df_p2_ = pd.read_csv('experiments/model2_1000k/p2.csv', header=None)
# df_p3_ = pd.read_csv('experiments/model2_1000k/p3.csv', header=None)
# b1 = list(df_b1_.loc[0])
# b2 = list(df_b2_.loc[0])
# p2 = list(df_p2_.loc[0])
# p3 = list(df_p3_.loc[0])

df_b_p = pd.read_csv('experiments/model2_1000k/1000k_11.05.2021_11.34.340/b_p_1000000.csv')
b1 = df_b_p['b1']
b2 = df_b_p['b2']
p2 = df_b_p['p2']
p3 = df_b_p['p3']
model = 2
Time = 1000000
def plot_b_per3(b1, b2, p2, p3, time, N0, folder_name, m, fs, lw, n0):
    fig, ax = plt.subplots()
    # n = int(time/10)
    n = n0
    nn = n
    N = int(N0/2)

    def func2(x, a, b, c):
        return a * x ** b + c

    def func(x, a, b):
        return a * x ** b

    ydata_f = b1
    xdata_f = range(len(ydata_f))
    ydata = ydata_f[N:]
    xdata = xdata_f[N:]
    plt.xscale('log')
    plt.yscale('log')
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.plot(xdata_f[n:], ydata_f[n:], 'm-', label=r'$\beta_1(t)$',  linewidth=lw)

    if m != 1:
        try:
            popt, pcov = curve_fit(func, xdata, ydata, bounds=([0.,0.], [5, 0.675]))
        except RuntimeError:
            popt, pcov = curve_fit(func, xdata, ydata, bounds=([0., 0., -10], [10., 10., 900]))

        plt.plot(xdata_f[n:], func(xdata_f[n:], *popt), 'm--', label=r'$\beta_1(t)=%5.2f t^{%5.2f}$' % tuple(popt), linewidth=lw)

        print("plotting b2")
        N = N0
        ydata_f = b2
        xdata_f = range(len(ydata_f))
        ydata = ydata_f[N:]
        xdata = xdata_f[N:]
        plt.plot(xdata_f[n:], ydata_f[n:], 'b-', label=r'$\beta_2(t)$',  linewidth=0.75)
        try:
            popt, pcov = curve_fit(func2, xdata, ydata)
        except RuntimeError:
            print("exception")
            popt, pcov = curve_fit(func2, xdata, ydata, bounds=([0., 0., -5000], [10., 2, 4000]))
        print(popt)
        plt.plot(xdata_f[n:], func2(xdata_f[n:], *popt), 'b--', label=r'$\beta_2(t)=%5.2f t^{%5.2f}%+5.1f$' % tuple(popt),  linewidth=lw)
        # try:
        #     popt, pcov = curve_fit(func, xdata, ydata)
        # except RuntimeError:
        #     print("exception")
        #     popt, pcov = curve_fit(func, xdata, ydata, bounds=([0., 0., -5000], [10., 2, 4000]))
        # print(popt)
        # plt.plot(xdata_f[n:], func(xdata_f[n:], *popt), 'b--', label=r'$\beta_2(t)=%5.2f t^{%5.2f}$' % tuple(popt),  linewidth=lw)

        # Constrain the optimization to the linear function
        # try:
        #     popt, pcov = curve_fit(func2, xdata, ydata, bounds=([0., 0., -np.inf], [1., 1., np.inf]))
        # except ValueError:
        #     popt, pcov = curve_fit(func2, xdata, ydata, bounds=([0., 0., -5000], [1., 1., 10]))
        #
        # plt.plot(xdata_f[nn+n:], func2(xdata_f[nn+n:], *popt), 'g--', label=r'$y=%5.2f x^{%5.2f}%+5.1f$' % tuple(popt),  linewidth=lw)

    if m == 1:
        try:
            popt, pcov = curve_fit(func, xdata, ydata)#, bounds=([0.,0., 1000], [2., 1, 1500]))
        except RuntimeError:
            popt, pcov = curve_fit(func, xdata, ydata, bounds=([0., 0., -10], [10., 10., 900]))
        if popt[1] > 1.05:
            try:
                popt, pcov = curve_fit(func2, xdata, ydata)#, bounds=([0.,0., 1000], [2., 1, 1500]))
            except RuntimeError:
                popt, pcov = curve_fit(func2, xdata, ydata, bounds=([0., 0., -10], [10., 10., 900]))

            plt.plot(xdata_f[n:], func2(xdata_f[n:], *popt), 'm--', label=r'$\beta_1(t)=%5.2f t^{%5.2f}%+5.1f$' % tuple(popt), linewidth=lw)
        else:
            plt.plot(xdata_f[n:], func(xdata_f[n:], *popt), 'm--', label=r'$\beta_1(t)=%5.2f t^{%5.2f}$' % tuple(popt), linewidth=lw)

    print("plotting p2")
    ydata = p2
    xdata = range(len(ydata))
    plt.plot(xdata[n:], ydata[n:], color='orange', linestyle='solid', label=r'$P_{2}(t)$',  linewidth=lw)
    popt, pcov = curve_fit(func, xdata, ydata)
    plt.plot(xdata[n:], func(xdata[n:], *popt), color='orange', linestyle='dashed', label=r'$P_2(t)=%5.2f t^{%5.2f}$' % tuple(popt),  linewidth=lw)
    print("plotting p3")
    ydata = p3
    xdata = range(len(ydata))
    plt.plot(xdata[n:], ydata[n:], color='deepskyblue', linestyle='solid', label=r'$P_{3}(t)$',  linewidth=lw)
    popt, pcov = curve_fit(func, xdata, ydata)
    plt.plot(xdata[n:], func(xdata[n:], *popt), color='deepskyblue', linestyle='dashed', label=r'$P_3(t)=%5.2f t^{%5.2f}$' % tuple(popt),  linewidth=lw)

    # font = {'size': fs}
    # plt.rc('font', **font)

    plt.rcParams.update({'font.size': fs, 'font.weight': 'light'})
    plt.xlabel('t', fontsize=fs)
    plt.ylabel('Growth Rates', fontsize=fs)
    handles, labels = ax.get_legend_handles_labels()
    # myorder = [0, 2, 4, 6, 1, 3, 5, 7]
    # handles = [handles[i] for i in myorder]
    # labels = [labels[i] for i in myorder]
    # plt.legend(handles, labels, prop={'size': 6}, loc='lower right', ncol=2)
    # plt.legend(handles, labels, prop={'size': fs-4}, loc='upper left', ncol=2)

    myorder1 = [4, 6, 5, 7]
    handles1 = [handles[i] for i in myorder1]
    labels1 = [labels[i] for i in myorder1]
    leg1 = ax.legend(handles1, labels1, prop={'size': fs-4}, loc='upper left', ncol=2)

    myorder2 = [0, 2, 1, 3]
    handles2 = [handles[i] for i in myorder2]
    labels2 = [labels[i] for i in myorder2]
    leg2 = ax.legend(handles2, labels2, prop={'size': fs-4}, loc='lower right', ncol=2)
    ax.add_artist(leg1)

    # plt.legend(loc=4, prop={'size': 6})
    # plt.rc('xtick', labelsize=fs)
    # plt.rc('ytick', labelsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.tick_params(axis='both', which='minor', labelsize=fs)
    # plt.rc('axes', labelsize=fs)
    # plt.rcParams.update({'font.size': fs, 'font.weight': 'light'})
    plt.tight_layout()
    plt.savefig(folder_name+'/per-b-time.jpg', dpi=100)
    plt.savefig(folder_name+'/per-b-time.pdf', dpi=400)
    print("\n done")
    plt.close()
print("Plotting the growth rates of Betti numbers and the perimeter...")
# plot_b_per3(Betti_1_total_vector, Betti_2_total_vector, Per_2d, Per_3d, Time, int(Time*0.5), folder_name, model, 18,1, 30000)
plot_b_per3(b1, b2, p2, p3, Time, 220000, 'experiments/model2_1000k', model, 18, 1, 50000)

"""FREQUENCIES BETTI"""
# b1
for i in range(df_b1.shape[0]):
    b1 = list(df_b1.loc[i])
    changes = [b1[i+1]-b1[i] for i in range(len(b1)-1)]
    counter = collections.Counter(changes)
    s = sum(counter.values())
    for key in counter:
        counter[key] /= s
    print(counter)
    with open('experiments/FREQ_B/MODEL 1'+'/b1_freq.csv', 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(counter.values())
fold = 'experiments/FREQ_B/df_all_500000.csv'
df_b_ = pd.read_csv(fold, low_memory=False)
dfs_b = [df_b_[:500000], df_b_[500001:1000001], df_b_[1000002:1500002], df_b_[1500003:2000003]]
for j in range(len(dfs_b)):
    b1 = list(dfs_b[j]['b1'])
    b1 = [int(x) for x in b1]
    changes = [b1[i+1]-b1[i] for i in range(len(b1)-1)]
    counter = collections.Counter(changes)
    s = sum(counter.values())
    for key in counter:
        counter[key] /= s
    print(j, counter)
    print(counter.values())
    if j < 3:
        with open('experiments/FREQ_B/MODEL 0'+'/b1_freq.csv', 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(counter.values())
    else:
        with open('experiments/FREQ_B/MODEL 1'+'/b1_freq.csv', 'a+') as f:
            writer = csv.writer(f)
            co = sorted(list(counter.values()), reverse=True)
            print(co)
            writer.writerow(co)

fold = 'experiments/model2_500k/model2_500k/b2.csv'
df_b_ = pd.read_csv(fold, low_memory=False, header=None)
print(df_b_.shape)
for j in range(3):
    b2 = df_b_.loc[j]
    changes = [b2[i+1]-b2[i] for i in range(len(b2)-1)]
    counter = collections.Counter(changes)
    s = sum(counter.values())
    for key in counter:
        counter[key] /= s
    counter = list(counter.values())
    counter = [counter[2], counter[0], counter[1]]
    print(counter)
    with open('experiments/FREQ_B/MODEL 2'+'/b2_freq.csv', 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(counter)

fold = 'experiments/FREQ_B/MODEL 2/b2_freq.csv'
b2_fr = pd.read_csv(fold, low_memory=False, header=None)

fold = 'experiments/FREQ_B/MODEL 0/b2_freq.csv'
df_b_ = pd.read_csv(fold, low_memory=False, header=None)

# b2
for i in range(df_b2_.shape[0]):
    b2 = list(df_b2_.loc[i])
    b2 = b2[:500000]
    changes = [b2[i+1]-b2[i] for i in range(len(b2)-1)]
    counter = collections.Counter(changes)
    s = sum(counter.values())
    for key in counter:
        counter[key] /= s
    print(counter)
    with open('experiments/FREQ_B/MODEL 0'+'/b2_freq.csv', 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(counter.values())

fold = 'experiments/FREQ_B/MODEL 2/df_freq_b2_500000.csv'
df_b_ = pd.read_csv(fold, low_memory=False)
b2 = list(df_b_['b2'])
# b2 = b2[:500000]
changes = [b2[i+1]-b2[i] for i in range(len(b2)-1)]
counter = collections.Counter(changes)
s = sum(counter.values())
for key in counter:
    counter[key] /= s
print(counter)
with open('experiments/FREQ_B/MODEL 2'+'/b2_freq.csv', 'a+') as f:
    writer = csv.writer(f)
    writer.writerow(list(df_b_.loc[500000])[1:])


#
# """Holes N clusters"""
# #created
# max_hole_size = max([list(Created_holes_array[i].keys())[-1] for i in range(num_models)])
# cr_h = np.zeros((num_models, max_hole_size))
# for i in range(num_models):
#     all_holes = sum(Created_holes_array[i].values())
#     for j in range(max_hole_size):
#         if j+1 not in Created_holes_array[i]:
#             cr_h[i][j] = 0
#         else:
#             cr_h[i][j] = Created_holes_array[i][j+1]/all_holes
# pd.DataFrame(cr_h, columns=np.arange(1, max_hole_size+1)).to_csv(folder_name+r'/df_holes_total_' + str(num_models) + '.csv', mode='w+', header=True)
#
# max_size = 5
# for i in range(num_models):
#     cr_h[i][max_size-1] += sum(cr_h[i][max_size:])
# pd.DataFrame([np.mean(cr_h[:, :max_size], axis=0), np.std(cr_h[:, :max_size], axis=0)],
#              columns=np.arange(1, max_size+1), index=['Mean', 'SD']).to_csv(folder_name+r'/df_holes_cr_statistics_'
#                                                                              + str(num_models) + '.csv', mode='w+', header=True)
# #final
# max_hole_size = max([list(Holes_array[i].keys())[-1] for i in range(num_models)])
# f_h = np.zeros((num_models, max_hole_size))
# for i in range(num_models):
#     all_holes = sum(Holes_array[i].values())
#     for j in range(max_hole_size):
#         if j+1 not in Holes_array[i]:
#             f_h[i][j] = 0
#         else:
#             f_h[i][j] = Holes_array[i][j+1]/all_holes
# pd.DataFrame(f_h, columns=np.arange(1, max_hole_size+1)).to_csv(folder_name+r'/df_holes_final_' + str(num_models) + '.csv', mode='w+', header=True)
# draw_diagram_holes2(np.mean(cr_h, axis=0), np.mean(f_h, axis=0), folder_name, max=10)
#
# for i in range(num_models):
#     f_h[i][max_size-1] += sum(f_h[i][max_size:])
# pd.DataFrame([np.mean(f_h[:, :max_size], axis=0), np.std(f_h[:, :max_size], axis=0)],
#              columns=np.arange(1, max_size+1), index=['Mean', 'SD']).to_csv(folder_name+r'/df_holes_final_statistics_'
#                                                                              + str(num_models) + '.csv', mode='w+', header=True)
# """FREQUENCIES"""
# Fr_b1_mean = dict()
# for change in list(Fr_b1_array[0].keys()):
#     Fr_b1_mean[change] = np.mean([Fr_b1_array[i][change] for i in range(num_models)], axis=0)
# draw_frequencies_1(Fr_b1_mean, folder_name)
#
# Fr_b2_mean = dict()
# for change in list(Fr_b2_array[0].keys()):
#     Fr_b2_mean[change] = np.mean([Fr_b2_array[i][change] for i in range(num_models)], axis=0)
# draw_frequencies_2(Fr_b2_mean, folder_name)
#
# b1_table = pd.DataFrame(columns=Fr_b1_array[0].keys())
# for ch in Fr_b1_array[0].keys():
#     b1_table[ch] = [np.mean([Fr_b1_array[i][ch][-1] for i in range(num_models)]), np.std([Fr_b1_array[i][ch][-1] for i in range(num_models)])]
# b1_table.index = ['Mean', 'SD']
# b1_table.to_csv(folder_name+r'/df_b1_fr_' + str(num_models) + '.csv', mode='w+', header=True)
#
# b2_table = pd.DataFrame(columns=Fr_b2_array[0].keys())
# for ch in Fr_b2_array[0].keys():
#     b2_table[ch] = [np.mean([Fr_b2_array[i][ch][-1] for i in range(num_models)]),np.std([Fr_b2_array[i][ch][-1] for i in range(num_models)])]
# b2_table.index = ['Mean', 'SD']
# b2_table.to_csv(folder_name+r'/df_b2_fr_' + str(num_models) + '.csv', mode='w+', header=True)
#
# """POLYCUBES"""
# Tricube = dict(pd.DataFrame([Polycubes[i][0] for i in range(num_models)]).mean())
# Tricube_f = dict(pd.DataFrame([Polycubes[i][1] for i in range(num_models)]).mean())
# Tetracube = dict(pd.DataFrame([Polycubes[i][2] for i in range(num_models)]).mean())
# Tetracube_f = dict(pd.DataFrame([Polycubes[i][3] for i in range(num_models)]).mean())
# draw_tri_tetra(Tricube, Tricube_f, Tetracube, Tetracube_f, folder_name)
"""FILE CASE"""
# if file:
#     print('What is the format of the file? \n0 -- list of tuples \n1 -- Perseus')
#     file_format = read_value([0, 1])
#     print('Name of the file (for example, filename.txt):')
#     filename = str(input())
#     while not Path(str(dim)+"d/files/"+filename).exists():
#         print("Oops!  That was no valid name.  Try again...")
#         filename = str(input())
#     if file_format == 1:
#         Eden_f = read_eden_perseus(str(dim)+"d/files/"+filename)
#     else:
#         Eden_f = read_eden_txt(str(dim)+"d/files/"+filename)
#     Eden = [x[0] for x in Eden_f]
#     Process = Eden.copy()
#     Times = [x[1] for x in Eden_f]
#     Time = len(Eden)
#     now = datetime.now()
#     dt_string = now.strftime("%d:%m:%Y_%H.%M.%S")
#     folder_name = filename
#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)
#
#     print('Do you want GUDHI barcode(s)? \n0 -- no \n1 -- yes')
#     gudhi = bool(read_value([0, 1]))
#
#     print("\nComputing persistent homology...")
#     Eden, Perimeter, Betti_2_total_vector, Betti_1_total_vector, Barcode, Holes, \
#         Betti_2_total, Betti_1_total, Created_holes, Perimeter_len, \
#         Final_barcode = grow_eden_debugging(len(Eden), Eden)
#
#     print("\nCalculating frequencies of Betti_1...")
#     freq, changes = return_frequencies_1(Betti_1_total_vector, Time)
#     draw_frequencies_1(freq, changes, folder_name)
#     print("\nCalculating frequencies of Betti_2...")
#     freq, changes = return_frequencies_2(Betti_2_total_vector, Time)
#     draw_frequencies_2(freq, changes, folder_name)
#
#     print("Plotting the frequency of the volume of top dimensional \"holes\"...")
#     draw_diagram_holes(Created_holes, Holes, folder_name)
#     print("Plotting the growth rates of Betti numbers and the perimeter...")
#     plot_b_per(Betti_1_total_vector, Betti_2_total_vector, Perimeter_len, Time, 0, folder_name)
#     print("Plotting the frequency of the number of top dimensional holes for specific shapes with 3 and 4 cells...")
#     Tricube, Tricube_f, Tetracube, Tetracube_f = num_holes(Created_holes, Holes)
#     draw_tri_tetra(Tricube, Tricube_f, Tetracube, Tetracube_f, folder_name)
#
#     if pic:
#         a = 1
#         f = open("3d/"+str(int(Time/1000))+"k/MAYA.txt", "w+")
#         f.write("import maya.cmds as cmds \nimport math as m \n"
#                 "import os,sys \nEden = " + str(Process)+"\nt = len(Eden)"
#                 "\nfor i in range(0,t):\n\taux = cmds.polyCube()"
#                 "\n\tcmds.move(Eden[i][0],Eden[i][1],Eden[i][2],aux)")
#         f.close()
#         print("We created txt file \"MAYA\" for you. Just copy paste its content to MAYA!")

# df.to_csv(r'experiments/perimeter/df_' + str(Time) + '.csv', mode='a+', header=False)
#
# df2 = pd.concat([df.mean(axis=0), df.std(axis=0)], axis=1)
# df2 = df2.rename(columns={0: "Mean", 1: "SD"})
# df2.to_csv(r'experiments/perimeter/df2_' + str(Time) + '.csv', mode='a+', header=False)

print("WE ARE DONE! CHECK THE FOLDER!")
plot_b_per2(Betti_1_total_vector, Betti_2_total_vector, Per_2d, Per_3d, Time, 0, folder_name, model, 14,1)

plot_b_per2(b1,b2,p2,p3, Time, 0, 'experiments', model, 20, 1)
def plot_b_per2(b1, b2, p2, p3, time, N, folder_name, m, fs, lw):
    fig, ax = plt.subplots()
    # n = int(time/10)
    n = 10
    nn = n
    N = 0

    def func2(x, a, b, c):
        return a * x ** b + c

    def func(x, a, b):
        return a * x ** b

    ydata_f = b1
    xdata_f = range(len(ydata_f))
    ydata = ydata_f[N:]
    xdata = xdata_f[N:]
    plt.xscale('log')
    plt.yscale('log')
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.plot(xdata_f[n:], ydata_f[n:], 'm-', label=r'$\beta_1(t)$',  linewidth=lw)

    if m != 1:
        try:
            popt, pcov = curve_fit(func, xdata, ydata)#, bounds=([0.,0., 1000], [2., 1, 1500]))
        except RuntimeError:
            popt, pcov = curve_fit(func, xdata, ydata, bounds=([0., 0., -10], [10., 10., 900]))

        plt.plot(xdata_f[n:], func(xdata_f[n:], *popt), 'm--', label=r'$\beta_1(t)=%5.2f t^{%5.2f}$' % tuple(popt), linewidth=lw)

        N = 350000
        ydata_f = b2
        xdata_f = range(len(ydata_f))
        ydata = ydata_f[N:]
        xdata = xdata_f[N:]
        plt.plot(xdata_f[n:], ydata_f[n:], 'b-', label=r'$\beta_2(t)$',  linewidth=0.75)
        try:
            popt, pcov = curve_fit(func2, xdata, ydata)
        except RuntimeError:
            popt, pcov = curve_fit(func2, xdata, ydata, bounds=([0., 0., -5000], [1., 2, 4000]))
        plt.plot(xdata_f[n:], func2(xdata_f[n:], *popt), 'b--', label=r'$\beta_2(t)=%5.2f t^{%5.2f}%+5.1f$' % tuple(popt),  linewidth=lw)

        # Constrain the optimization to the linear function
        # try:
        #     popt, pcov = curve_fit(func2, xdata, ydata, bounds=([0., 0., -np.inf], [1., 1., np.inf]))
        # except ValueError:
        #     popt, pcov = curve_fit(func2, xdata, ydata, bounds=([0., 0., -5000], [1., 1., 10]))
        #
        # plt.plot(xdata_f[nn+n:], func2(xdata_f[nn+n:], *popt), 'g--', label=r'$y=%5.2f x^{%5.2f}%+5.1f$' % tuple(popt),  linewidth=lw)

    if m == 1:
        try:
            popt, pcov = curve_fit(func, xdata, ydata)#, bounds=([0.,0., 1000], [2., 1, 1500]))
        except RuntimeError:
            popt, pcov = curve_fit(func, xdata, ydata, bounds=([0., 0., -10], [10., 10., 900]))
        if popt[1] > 1.05:
            try:
                popt, pcov = curve_fit(func2, xdata, ydata)#, bounds=([0.,0., 1000], [2., 1, 1500]))
            except RuntimeError:
                popt, pcov = curve_fit(func2, xdata, ydata, bounds=([0., 0., -10], [10., 10., 900]))

            plt.plot(xdata_f[n:], func2(xdata_f[n:], *popt), 'm--', label=r'$\beta_1(t)=%5.2f t^{%5.2f}%+5.1f$' % tuple(popt), linewidth=lw)
        else:
            plt.plot(xdata_f[n:], func(xdata_f[n:], *popt), 'm--', label=r'$\beta_1(t)=%5.2f t^{%5.2f}$' % tuple(popt), linewidth=lw)

    ydata = p2
    xdata = range(len(ydata))
    plt.plot(xdata[n:], ydata[n:], color='orange', linestyle='solid', label=r'$P_{2}(t)$',  linewidth=lw)
    popt, pcov = curve_fit(func, xdata, ydata)
    plt.plot(xdata[n:], func(xdata[n:], *popt), color='orange', linestyle='dashed', label=r'$P_2(t)=%5.2f t^{%5.2f}$' % tuple(popt),  linewidth=lw)
    ydata = p3
    xdata = range(len(ydata))
    plt.plot(xdata[n:], ydata[n:], color='deepskyblue', linestyle='solid', label=r'$P_{3}(t)$',  linewidth=lw)
    popt, pcov = curve_fit(func, xdata, ydata)
    plt.plot(xdata[n:], func(xdata[n:], *popt), color='deepskyblue', linestyle='dashed', label=r'$P_3(t)=%5.2f t^{%5.2f}$' % tuple(popt),  linewidth=lw)

    # font = {'size': fs}
    # plt.rc('font', **font)

    plt.rcParams.update({'font.size': fs, 'font.weight': 'light'})
    plt.xlabel('t', fontsize=fs)
    plt.ylabel('Growth Rates', fontsize=fs)
    handles, labels = ax.get_legend_handles_labels()
    myorder = [0, 2, 4, 6, 1, 3, 5, 8]
    handles = [handles[i] for i in myorder]
    labels = [labels[i] for i in myorder]
    # plt.legend(handles, labels, prop={'size': 6}, loc='lower right', ncol=2)
    plt.legend(handles, labels, prop={'size': fs-4}, loc='lower right', ncol=2)
    # plt.legend(loc=4, prop={'size': 6})
    # plt.rc('xtick', labelsize=fs)
    # plt.rc('ytick', labelsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.tick_params(axis='both', which='minor', labelsize=fs)
    # plt.rc('axes', labelsize=fs)
    # plt.rcParams.update({'font.size': fs, 'font.weight': 'light'})
    plt.tight_layout()
    plt.savefig(folder_name+'/per-b-time.jpg', dpi=100)
    plt.savefig(folder_name+'/per-b-time.pdf', dpi=400)
    plt.close()

plot_b_per2(df_b1_.mean(), df_b2_.mean(), df_p2_.mean(), df_p3_.mean(), Time, int(Time/3), 'experiments', model, 8, 1)
plot_b_per(Betti_1_total_vector, Betti_2_total_vector, Per_2d, Per_3d, Time, 0, folder_name, model)

fold = 'experiments/FREQ_B/MODEL 2/df_freq_b1_500000.csv'
df_b_ = pd.read_csv(fold, low_memory=False)
dict_f = {0: df_b_['0'], 1: df_b_['1'], -1: df_b_['-1'], 2: df_b_['2']}
draw_frequencies_1_2(dict_f,'experiments', 20)

# fold = 'experiments/FREQ_B/MODEL 0/df_freq_b2_500000.csv'
# df_b_ = pd.read_csv(fold, low_memory=False)
# dict_f = {-1: df_b_['-1'],0: df_b_['0'], 1: df_b_['1']}
# draw_frequencies_2_2(dict_f, 'experiments', 22)

def draw_frequencies_2_2(dict, folder_name, fs):
    fig, ax = plt.subplots()
    l = len(dict[0])
    lw = 1
    sh = []
    for j in np.arange(-1, 2):
        sh.append(next((i for i, x in enumerate(dict[j]) if x), 0))
    shift = max(sh)
    mean_values = {x: sum(dict[x][(-int(len(dict[x])/10)):]) / len(dict[x][(-int(len(dict[x])/10)):]) for x in range(0, 2)}

    if next((i for i, x in enumerate(dict[-1]) if x), 0) != 0:
        mean_values = {x: sum(dict[x][(-int(len(dict[x])/10)):]) / len(dict[x][(-int(len(dict[x])/10)):]) for x in range(-1, 2)}
        ax.plot(range(shift, l), dict[-1][shift:], color='tab:red', label='-1', linewidth=lw)
        ax.plot(range(shift, l), [mean_values[-1]]*len(range(shift, l)), color='tab:red', linestyle='--', linewidth=lw)
    ax.plot(range(shift, l), dict[0][shift:], color='tab:orange', label='0', linewidth=lw)
    ax.plot(range(shift, l), dict[1][shift:], color='tab:green', label='+1', linewidth=lw)

    ax.plot(range(shift, l), [mean_values[0]]*len(range(shift, l)), color='tab:orange', linestyle='--', linewidth=lw)
    ax.plot(range(shift, l), [mean_values[1]]*len(range(shift, l)), color='tab:green', linestyle='--', linewidth=lw)

    plt.yscale('log')

    plt.rcParams.update({'font.size': fs, 'font.weight': 'light'})
    ax.set_ylabel(r'Frequency of Change in $\beta_2$')
    ax.set_xlabel('t')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    my_yticks = list(mean_values.values())
    my_yticks2 = [round(x, 3) for x in my_yticks]
    plt.tight_layout()
    plt.yticks(my_yticks, my_yticks2)
    plt.legend(loc="lower right", prop={'size': fs})
    fig.savefig(folder_name+'/fr_b_2_f.jpg', dpi=100)
    fig.savefig(folder_name+'/fr_b_2_f.pdf', dpi=500)
    plt.close()
def draw_frequencies_1_2(dict, folder_name, fs):
    print("\nPlotting frequencies of Betti_1...")
    fig, ax = plt.subplots()
    l = len(dict[0])

    sh = []
    for j in np.arange(-1, 3):
        sh.append(next((i for i, x in enumerate(dict[j]) if x), 0))
    shift = max(sh)*10
    mean_values = {x: sum(dict[x][(-int(len(dict[x])/10)):]) / len(dict[x][(-int(len(dict[x])/10)):]) for x in range(-1, 3)}

    linew = 1
    ax.plot(range(shift, l), dict[-1][shift:], color='tab:red', label='-1',  linewidth=linew)
    ax.plot(range(shift, l), dict[0][shift:], color='tab:orange', label='0',  linewidth=linew)
    ax.plot(range(shift, l), dict[1][shift:], color='tab:green', label='+1',  linewidth=linew)
    ax.plot(range(shift, l), dict[2][shift:], color='tab:blue', label='+2',  linewidth=linew)

    ax.plot(range(shift, l), [mean_values[-1]]*len(range(shift, l)), color='tab:red', linestyle='--', linewidth=linew)
    ax.plot(range(shift, l), [mean_values[0]]*len(range(shift, l)), color='tab:orange', linestyle='--', linewidth=linew)
    ax.plot(range(shift, l), [mean_values[1]]*len(range(shift, l)), color='tab:green', linestyle='--', linewidth=linew)
    ax.plot(range(shift, l), [mean_values[2]]*len(range(shift, l)), color='tab:blue', linestyle='--', linewidth=linew)

    plt.yscale('log')
    plt.rcParams.update({'font.size': fs, 'font.weight': 'light'})
    ax.set_ylabel(r'Frequency of Change in $\beta_1$')
    ax.set_xlabel('t')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    my_yticks = list(mean_values.values())
    my_yticks2 = [round(x, 3) for x in my_yticks]
    my_yticks[0] += 0.02
    my_yticks[2] -= 0.02
    plt.yticks(my_yticks, my_yticks2)
    ax.tick_params(axis='y', which='major', labelsize=fs)
    ax.tick_params(axis='y', which='minor', labelsize=fs)
    # ax.legend(loc=1, prop={'size': 6})

    plt.tight_layout()
    ax.legend(loc ='lower right', bbox_to_anchor=(1, 0.15), ncol =2)
    fig.savefig(folder_name+'/fr_b_1.jpg', dpi=100)
    fig.savefig(folder_name+'/fr_b_1.pdf', dpi=500)
    plt.close()
def draw_frequencies_1_eu2(dict, changes, folder_name, fs):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    l = len(dict[0])

    # ch_1 = [i for i, j in enumerate(changes) if j == -1]
    # y_1 = []
    # for x in ch_1:
    #     y_1 += [dict[1][x+1]]

    sh = []
    for j in np.arange(0, 3):
        sh.append(next((i for i, x in enumerate(dict[j]) if x), 0))
    shift = max(sh)*5

    linew = 1

    ax.plot(range(shift, l), dict[0][shift:], color='tab:orange', label='0',  linewidth=linew)
    ax.plot(range(shift, l), dict[1][shift:], color='tab:green', label='+1',  linewidth=linew)
    ax.plot(range(shift, l), dict[2][shift:], color='tab:blue', label='+2',  linewidth=linew)

    mean_values = {x: sum(dict[x][(-int(len(dict[x])/10)):]) / len(dict[x][(-int(len(dict[x])/10)):]) for x in range(0, 3)}

    ax.plot(range(shift, l), [mean_values[0]]*len(range(shift, l)), color='tab:orange', linestyle='--', linewidth=linew)
    ax.plot(range(shift, l), [mean_values[1]]*len(range(shift, l)), color='tab:green', linestyle='--', linewidth=linew)
    ax.plot(range(shift, l), [mean_values[2]]*len(range(shift, l)), color='tab:blue', linestyle='--', linewidth=linew)

    # if next((i for i, x in enumerate(dict[-1]) if x), 0) != 0:
    #     plt.scatter(ch_1, y_1, s=5, marker='o', color="tab:red", label='-1')

    plt.rcParams.update({'font.size': fs, 'font.weight': 'light'})
    plt.yscale('log')
    ax.set_ylabel(r'Frequency of Change in $\beta_1$')
    ax.set_xlabel('t')

    my_yticks = list(mean_values.values())
    my_yticks2 = [round(x, 3) for x in my_yticks]
    plt.yticks(my_yticks, my_yticks2)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # ax.legend(loc=1, prop={'size': 6})
    ax.legend(loc ='lower right', bbox_to_anchor=(1, 0.2))
    plt.tight_layout()
    fig.savefig(folder_name+'/fr_b_1_eu.jpg', dpi=100)
    fig.savefig(folder_name+'/fr_b_1_eu.pdf', dpi=500)
    plt.close()


# def draw_diagram_holes2(created_holes, holes, folder_name):
#     fr_cr = [created_holes[i][-2] for i in range(len(created_holes))]
#     fr_cr.sort()
#     fr_final = [len(holes[i]) for i in holes]
#     fr_final.sort()
#     width = 0.5
#
#     def func(x, a, b):
#         return a * x ** b
#
#     counter_cr = collections.Counter(fr_cr)
#     counter_cr2 = {k: v/sum(counter_cr.values()) for (k, v) in counter_cr.items()}
#     num = 11
#     num_plus = 0
#     for key in list(counter_cr2.keys()):
#         if key >= num:
#             # num_plus += counter_cr2[key]
#             del counter_cr2[key]
#     # counter_cr2[num] = num_plus
#
#     xdata = list(counter_cr2.keys())
#     ydata = list(counter_cr2.values())
#
#     try:
#         popt, pcov = curve_fit(func, xdata, ydata)#, bounds=([0.,0., 1000], [2., 1, 1500]))
#     except RuntimeError:
#         popt, pcov = curve_fit(func, xdata, ydata, bounds=([0., -10.], [10000., 10.]))
#
#     fig, ax = plt.subplots()
#     plt.yscale('log')
#
#     xdataplot = np.arange(xdata[0], xdata[-1], 0.1)
#     # # print(ydata, list(func(xdataplot, *popt)))
#     #
#     plt.plot(xdataplot-width/2, list(func(xdataplot, *popt)), color=[0, 94/255, 255/255], label=r'fit: $y=%5.2f x^{%5.3f}$' % tuple(popt), linewidth=0.75)
#     # # plt.show()
#
#     # xdata = np.array(xdata)
#     # ydata = np.array(ydata)
#     # log_x_data = np.log(xdata)
#     # log_ydata = np.log(ydata)
#     # fit = np.polyfit(xdata, log_y_data, 1)
#     # print(xdata, log_y_data)
#     # print(fit)
#     # y = np.exp(fit[0]) * np.exp(fit[1]*xdata)
#     # plt.plot(xdata, y,  label=r'fit: $y=%5.2f e^{%5.3fx}$' % tuple(fit), linewidth=0.75)
#
#     # try:
#     #     popt, pcov = curve_fit(func2, xdata, log_ydata)#, bounds=([0.,0., 1000], [2., 1, 1500]))
#     # except RuntimeError:
#     #     popt, pcov = curve_fit(func2, xdata, log_ydata, bounds=([-10., -10.], [10000., 10.]))
#
#     # popt[0] = int(popt[0])
#     # print(popt)
#     # print(type(popt[0]))
#
#     # y_plot = np.exp(popt[0]) * np.exp(-popt[1]*xdataplot)
#
#     # plt.plot(xdataplot-width/2, y_plot, color=[1, 0.27, 0.95], label=r'fit: $y=%5.2f e^{-%5.3f*x}$' % tuple([np.exp(popt[0]), popt[1]]), linewidth=0.75)
#
#     for j in range(1, list(counter_cr.keys())[-1]):
#         if j not in counter_cr:
#             counter_cr[j] = 0
#
#     counter_final = collections.Counter(fr_final)
#     for i in counter_cr.keys():
#         if i not in counter_final:
#             counter_final[i] = 0
#     counter_final2 = {k: v/sum(counter_final.values()) for (k, v) in counter_final.items()}
#     num_plus = 0
#     for key in list(counter_final2.keys()):
#         if key >= num:
#             # num_plus += counter_final2[key]
#             del counter_final2[key]
#     # counter_final2[num] = num_plus
#
#     labels = range(len(counter_cr2.keys())+1)
#     x = np.arange(len(labels))
#
#     ax.bar(np.array(list(counter_cr2.keys())) - width/2, counter_cr2.values(), width, color=[(0.44, 0.57, 0.79)], label='Total')
#
#     xdata = list(counter_final2.keys())
#     ydata = list(counter_final2.values())
#
#     try:
#         popt, pcov = curve_fit(func, xdata, ydata)#, bounds=([0.,0., 1000], [2., 1, 1500]))
#     except RuntimeError:
#         popt, pcov = curve_fit(func, xdata, ydata, bounds=([0., -10.], [10000., 10.]))
#
#     xdataplot = np.arange(xdata[0], xdata[-1], 0.1)
#     plt.plot(xdataplot+width/2, list(func(xdataplot, *popt)), color=(225/256, 128/256, 0/256), label=r'fit: $y=%5.2f x^{%5.3f}$' % tuple(popt), linewidth=0.75)
#
#     ax.bar(np.array(list(counter_final2.keys())) + width/2, counter_final2.values(), width, color=[(225/256, 174/256, 122/256)], label='Final')
#
#     ax.set_ylabel('Frequency of Number of Holes')
#     ax.set_xlabel('Volume')
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels)
#     if len(labels) >= 70:
#         plt.setp(ax.get_xticklabels(), fontsize=2)
#     elif len(labels) >= 60:
#         plt.setp(ax.get_xticklabels(), fontsize=3)
#     elif len(labels) >= 50:
#         plt.setp(ax.get_xticklabels(), fontsize=4)
#     elif len(labels) >= 45:
#         plt.setp(ax.get_xticklabels(), fontsize=5)
#     elif len(labels) >= 40:
#         plt.setp(ax.get_xticklabels(), fontsize=6)
#     elif len(labels) >= 30:
#         plt.setp(ax.get_xticklabels(), fontsize=7)
#     elif len(labels) >= 20:
#         plt.setp(ax.get_xticklabels(), fontsize=6)
#
#     ax.legend(loc=1)
#     fig.tight_layout()
#     fig.savefig(folder_name+'/holes.png', format='png', dpi=500)
#     fig.savefig(folder_name+'/holes.pdf')
#     plt.close()
# draw_diagram_holes2(Created_holes, Holes, folder_name)
fold = 'experiments/Per_2d_in_array.csv'
p2 = pd.read_csv(fold, low_memory=False)
p2 = p2['4']
p2_M = [0.13316445293746645,0.1374828402021205, 0.14034515589741284,0.14381249166207724,0.1461184031232769,0.1479369735970474]
times = list(range(0,500000,50000))
p2 = [x for i,x in enumerate(p2) if i in times]
p2=list(p2)+p2_M[1:]

fold = 'experiments/Per_3d_in_array.csv'
p3 = pd.read_csv(fold, low_memory=False)
p3 = p3['10']
p3_M = [0.6023231320380067,	0.6169908551049454,	0.630494152190436,	0.6418239205584954,	0.6510198182446295]
times = list(range(0,500000,50000))
p3 = [x for i,x in enumerate(p3) if i in times]
p3=list(p3)+p3_M

def plot_per_inner2(p2, p3, time, folder_name,fs):
    n = int(0.25*len(p2))
    from scipy.optimize import curve_fit

    def func(x, a, b):
        return a * x ** b

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # plt.yscale('log')
    linew = 1

    ydata = p3
    xdata = list(range(0,500000,50000))+list(range(600000,1000001,100000))
    plt.plot(xdata, ydata, color='forestgreen', linestyle='solid', label=r'inner $P_{3}(t)$ data',  linewidth=linew)
    # popt, pcov = curve_fit(func, xdata[n:], ydata[n:])
    # plt.plot(xdata[n:], func(xdata[n:], *popt), color='forestgreen', linestyle='dashed', label=r'fit: $y=%5.4f x^{%5.3f}$' % tuple(popt),  linewidth=linew)

    ydata = p2
    xdata = list(range(0,500000,50000))+list(range(600000,1000001,100000))
    plt.plot(xdata, ydata, color='mediumorchid', linestyle='solid', label=r'inner $P_{2}(t)$ data',  linewidth=linew)
    # popt, pcov = curve_fit(func, xdata[n:], ydata[n:])
    # plt.plot(xdata[n:], func(xdata[n:], *popt), color='mediumorchid', linestyle='dashed', label=r'fit: $y=%5.6f x^{%5.3f}$' % tuple(popt),  linewidth=linew)

    mean_p2 = sum(p2[-int(len(p2)/10):])/len(p2[-int(len(p2)/10):])
    # mean_p3 = sum(p3[-int(len(p3)/20):])/len(p3[-int(len(p3)/20):])
    #
    # plt.plot(range(time), [mean_p3]*time, color='forestgreen', linestyle='--', linewidth=0.75)
    plt.rcParams.update({'font.size': fs, 'font.weight': 'light'})
    plt.plot(range(time), [p2[-1]]*time, color='mediumorchid', linestyle='--', linewidth=linew)
    plt.tight_layout()
    plt.xlabel('t', fontsize=fs)
    plt.ylabel('Fraction of the Perimeter',fontsize=14)
    plt.legend(loc='right', prop={'size': fs})
    # my_yticks = [mean_p2, mean_p3]
    # my_yticks2 = [round(x, 3) for x in my_yticks]
    # plt.yticks(my_yticks, my_yticks2)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(folder_name+'/per-inner.jpg', dpi=80)
    plt.savefig(folder_name+'/per-inner.pdf', dpi=500)
    plt.close()
plot_per_inner2(p2, p3, 1000000, 'experiments', 14)

fold = 'experiments/holes_final_10.csv'
polyc_f = pd.read_csv(fold, low_memory=False)
p = np.array(polyc_f,dtype=float)
for i in p:
    s = i[1]+i[2]
    i[1] = i[1]/s
    i[2] = float(i[2])/s
    s = sum(i[3:])
    for j in range(len(i[3:])):
        i[3+j] /= s
final_polycube = [round(x, 3) for x in p.mean(axis=0)]
columns = list(polyc_f.columns)
Tricube_f = dict(zip(columns[1:3], final_polycube[1:3]))
Tetracube_f = dict(zip(columns[3:], final_polycube[3:]))

fold = 'experiments/holes_total_10.csv'
polyc = pd.read_csv(fold, low_memory=False)
p = np.array(polyc,dtype=float)
for i in p:
    s = i[1]+i[2]
    i[1] = i[1]/s
    i[2] = float(i[2])/s
    s = sum(i[3:])
    for j in range(len(i[3:])):
        i[3+j] /= s
total_polycube = [round(x, 3) for x in p.mean(axis=0)]
Tricube = dict(zip(columns[1:3], total_polycube[1:3]))
Tetracube = dict(zip(columns[3:], total_polycube[3:]))
def draw_tri_tetra2(tri, tri_f, tetra, tetra_f, folder_name,fs):
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
        plt.rcParams.update({'font.size': fs, 'font.weight': 'light'})
        ax.set_ylabel('Number of Polycubes')
        ax.set_xlabel('Type of a Polycube')
        ax.set_xticks(x)
        ax.set_xticklabels(labels,fontsize = 12)
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder_name+'/tri-tetra-cubes.pdf', dpi=500)

        fig.savefig(folder_name+'/tri-tetra-cubes.jpg', dpi=70)
        plt.close()
    except ValueError:
        print("No tricubes and tetracubes in this complex")
        plt.close()
draw_tri_tetra2(Tricube, Tricube_f, Tetracube, Tetracube_f, 'experiments',12)

total = list(polyc.mean(axis=0))
final = list(polyc_f.mean(axis=0))
columns = polyc.columns
Tricube = dict(zip(columns[1:3], total[1:3]))
Tetracube = dict(zip(columns[3:], total[3:]))
Tricube_f = dict(zip(columns[1:3], final[1:3]))
Tetracube_f = dict(zip(columns[3:], final[3:]))
draw_tri_tetra2(Tricube, Tricube_f, Tetracube, Tetracube_f, 'experiments', 12)

fold = 'experiments/model1_500k/model1_500k/500k_11.05.2021_09.18.111/b_p_500000.csv'
df = pd.read_csv(fold)
b1 = df['b1'].tolist()
changes = [b1[i+1]-b1[i] for i in range(len(b1)-1)]
counter = collections.Counter(changes)
s = sum(counter.values())
for key in counter:
    counter[key] /= s
print(counter)
with open('experiments/FREQ_B/MODEL 1'+'/b1_freq.csv', 'a+') as f:
    writer = csv.writer(f)
    writer.writerow(counter.values())

fold = 'experiments/FREQ_B/MODEL 1'+'/b1_freq.csv'
df = pd.read_csv(fold, header=None)
