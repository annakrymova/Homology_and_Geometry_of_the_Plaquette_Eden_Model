import os
from datetime import datetime
import gudhi as gd
import matplotlib.pyplot as plt


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
file = bool(read_value([0, 1]))
# file = bool(0)

print('Which type of model? \n0 -- standard 2d in 3d growth model \n1 -- euler-characteristic mediated 2d in 3d growth model'
      '\n2 -- filled-in cubes 2d in 3d \n3 -- euler-characteristic mediated 3d growth model')
model = bool(read_value([0, 1, 2, 3]))
# model = 0

print('Do you want a picture of your model? (with a large model it can take time)  \n0 -- no \n1 -- yes')
pic = bool(read_value([0, 1]))
# pic = bool(0)

if pic:
    print('Do you want Python or MAYA 3D model? (We wouldn\'t recommend Python for large models (more than 500 tiles)).'
          ' \n0 -- Python \n1 -- MAYA')
    maya = bool(read_value([0, 1]))
    # maya = bool(0)

"""NO FILE CASE"""
if not file:
    print('How many tiles would you like in your model?')
    while True:
        try:
            Time = int(input())
            break
        except ValueError:
            print("Oops!  That was no valid number.  Try again...")
    # Time = 15000

    print('How many models would you like to build?')
    while True:
        try:
            num_models = int(input())
            break
        except ValueError:
            print("Oops!  That was no valid number.  Try again...")
    # num_models = 1

    for q in range(num_models):
        print("WORKING ON MODEL #"+str(q+1))
        now = datetime.now()
        dt_string = now.strftime("%d:%m:%Y_%H.%M.%S")
        if Time >= 1000:
            t = int(Time/1000)
            folder_name = str(t)+'k_'+dt_string
        else:
            t = Time
            folder_name = str(t)+'_'+dt_string
        folder_name = 'experiments/'+folder_name
        os.makedirs(folder_name)

        if model == 0 or model == 1 or model == 2:
            from e_2d_in_3d import num_holes, grow_eden, return_frequencies_1, return_frequencies_2, draw_barcode, \
                draw_frequencies_1, draw_frequencies_2, draw_diagram_holes, draw_tri_tetra, plot_b_per, draw_eden, \
                draw_frequencies_1_eu, draw_frequencies_2_eu, draw_barcode_gudhi, create_dist_matrix, plot_per_inner,\
                neighbours_diag, read_barcode_b1_from_file

            """BARCODE FOR B1"""
            # folder_name = '.'
            # Barcode_b1 = read_barcode_b1_from_file(folder_name)
            # Barcode_b1 = [a for a in Barcode_b1 if a[1][1] - a[1][0] != float('inf')]
            # draw_barcode_gudhi(Barcode_b1, folder_name, 1)

            print("Building a model...")
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
                Vertices, Process_ripser, Inner_perimeter, Voids, Per_2d_in, Per_3d_in = grow_eden(Time, model, folder_name)

            def plot_per_inner2(p2, p3, time, folder_name):
                def func(x, a, b):
                    return a * x ** b

                plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

                linew = 1

                ydata = p3
                xdata = range(len(ydata))
                plt.plot(xdata, ydata, color='forestgreen', linestyle='solid', label=r'inner $P_{3}(t)$ data',  linewidth=linew)
                # popt, pcov = curve_fit(func, xdata, ydata)
                # plt.plot(xdata, func(xdata, *popt), color='lightgreen', linestyle='dashed', label=r'fit: $y=%5.4f x^{%5.3f}$' % tuple(popt),  linewidth=linew)

                ydata = p2
                xdata = range(len(ydata))
                plt.plot(xdata, ydata, color='mediumorchid', linestyle='solid', label=r'inner $P_{2}(t)$ data',  linewidth=linew)
                # popt, pcov = curve_fit(func, xdata, ydata)
                # plt.plot(xdata, func(xdata, *popt), color='mediumorchid', linestyle='dashed', label=r'fit: $y=%5.6f x^{%5.3f}$' % tuple(popt),  linewidth=linew)
                part = 30
                mean_p2 = sum(p2[-int(len(p2)/part):])/len(p2[-int(len(p2)/part):])
                mean_p3 = sum(p3[-int(len(p3)/part):])/len(p3[-int(len(p3)/part):])

                plt.plot(range(time), [mean_p3]*time, color='forestgreen', linestyle='--', linewidth=0.75)
                plt.plot(range(time), [mean_p2]*time, color='mediumorchid', linestyle='--', linewidth=0.75)

                plt.xlabel('t')
                plt.ylabel('Fraction of the Perimeter')
                plt.legend(loc=4, prop={'size': 6})
                # plt.tight_layout()
                my_yticks = [mean_p2, mean_p3]
                my_yticks2 = [round(x, 3) for x in my_yticks]
                plt.yticks(my_yticks, my_yticks2)
                plt.savefig(folder_name+'/per-inner.png', dpi=500)
                plt.savefig(folder_name+'/per-inner.pdf', dpi=500)
                plt.close()
            plot_per_inner2(Per_2d_in, Per_3d_in, Time, folder_name)

            """DF b1 b2 per2 per3"""
            cols = ['b1', 'b2', 'p2', 'p3']
            df_all = pd.DataFrame(columns=cols)
            df_all['b1'] = Betti_1_total_vector
            df_all['b2'] = Betti_1_total_vector
            df_all['p2'] = Per_2d
            df_all['p3'] = Per_3d
            df_all.to_csv(r'CSV_filled_in/df_all_' + str(Time) + '.csv', mode='a+', header=True)

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
            plot_per_inner(Per_2d_in, Per_3d_in, Time, folder_name)
            #
            # """PERIMETER STATISTICS"""
            # per = [Per_2d[-1], len(Inner_perimeter) / Per_2d[-1], (Per_2d[-1] - len(Inner_perimeter)) / Per_2d[-1],
            #        Per_3d[-1], len([x for x in Voids if Voids[x][2] != 0]) / Per_3d[-1],
            #        len([x for x in Voids if Voids[x][2] == 0]) / Per_3d[-1]]
            #
            # df = df.append(dict(zip(cols, per)), ignore_index=True)

            create_dist_matrix(Time, Process_ripser, Vertices, folder_name)
            # print(Process)
            # freq, changes = return_frequencies_1(Betti_1_total_vector, Time)
            # if model == 1:
            #     draw_frequencies_1_eu(freq, changes, folder_name)
            # else:
            #     draw_frequencies_1(freq, folder_name)
            #
            # """DF freq b1 """
            # cols = list(freq.keys())
            # df = pd.DataFrame(columns=cols)
            # for col in cols:
            #     df[col] = freq[col]
            # df.to_csv(folder_name+r'/df_freq_b1_' + str(Time) + '.csv', mode='w+', header=True)
            #
            # freq, changes = return_frequencies_2(Betti_2_total_vector, Time)
            # if model == 1:
            #     draw_frequencies_2_eu(freq, changes, folder_name)
            # else:
            #     draw_frequencies_2(freq, changes, folder_name)
            #
            # """DF freq b2"""
            # cols = list(freq.keys())
            # df = pd.DataFrame(columns=cols)
            # for col in cols:
            #     df[col] = freq[col]
            # df.to_csv(folder_name+r'/df_freq_b2_' + str(Time) + '.csv', mode='w+', header=True)

            print("Plotting the frequency of the volume of top dimensional \"holes\"...")
            draw_diagram_holes(Created_holes, Holes, folder_name)
            print("Plotting the growth rates of Betti numbers and the perimeter...")
            plot_b_per(Betti_1_total_vector, Betti_2_total_vector, Per_2d, Per_3d, Time, int(Time/3), folder_name)
            print("Plotting the frequency of the number of top dimensional holes for specific shapes with 3 and 4 cells...")
            Tricube, Tricube_f, Tetracube, Tetracube_f = num_holes(Created_holes, Holes)
            draw_tri_tetra(Tricube, Tricube_f, Tetracube, Tetracube_f, folder_name)
            print("Plotting Betti_2 Barcode...")
            brc = [a[1] for a in Barcode]
            draw_barcode(brc, Time, folder_name)
            draw_barcode_gudhi(Barcode, folder_name)

            if pic:
                if maya:
                    f = open(folder_name+"/MAYA.txt", "w+")
                    f.write("import maya.cmds as cmds \n"
                            "Eden = " + str(Process)+"\nt = len(Eden)"
                            "\nfor i in range(t):"
                            "\n\tcmds.polyCreateFacet(p = Eden[i])")
                    f.close()
                    print("We created txt file \"MAYA\" for you. Just copy paste its content to MAYA!")
                else:
                    draw_eden(Eden, folder_name)
                    print("Python 3D model is created!")

        else:
            from e_3d import grow_eden, return_frequencies_1, return_frequencies_2, plot_b_per, draw_frequencies_1,\
                draw_frequencies_2
            from e_2d_in_3d import draw_diagram_holes, num_holes, draw_tri_tetra, draw_frequencies_2_eu
            Eden, Perimeter, Betti_2_total_vector, Betti_2_vector_changes, Barcode, Holes, Betti_1_total, \
                Betti_1_total_vector, Created_holes, Process, Perimeter_len, Skipped, I, Final_barcode = grow_eden(Time, model)

            print("\nCalculating frequencies of Betti_1...")
            freq, changes = return_frequencies_1(Betti_1_total_vector, Time)
            draw_frequencies_1(freq, changes, folder_name)
            print("\nCalculating frequencies of Betti_2...")
            freq, changes = return_frequencies_2(Betti_2_total_vector, Time)
            draw_frequencies_2(freq, changes, folder_name)
            draw_frequencies_2_eu(freq, changes, folder_name)

            if Created_holes and Holes:
                print("Plotting the frequency of the volume of top dimensional \"holes\"...")
                draw_diagram_holes(Created_holes, Holes, folder_name)
            print("Plotting the growth rates of Betti numbers and the perimeter...")
            plot_b_per(Betti_1_total_vector, Betti_2_total_vector, Perimeter_len, Time, 0, folder_name)
            print("Plotting the frequency of the number of top dimensional holes for specific shapes with 3 and 4 cells...")
            Tricube, Tricube_f, Tetracube, Tetracube_f = num_holes(Created_holes, Holes)
            draw_tri_tetra(Tricube, Tricube_f, Tetracube, Tetracube_f, folder_name)
            a = 10

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

print("WE ARE DONE! CHECK THE FOLDER!")



