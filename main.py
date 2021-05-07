from e_2d_in_3d import num_holes, grow_eden, return_frequencies_1, return_frequencies_2, draw_frequencies_1,\
            draw_frequencies_2, draw_diagram_holes, draw_tri_tetra, plot_b_per, draw_eden, draw_frequencies_1_eu,\
            draw_barcode_gudhi, create_dist_matrix, plot_per_inner, read_barcode_b1_from_file,  draw_pers_diagram,\
            create_directory, draw_diagram_holes2


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

def read_value_int():
    while True:
        try:
            x = int(input())
            break
        except ValueError:
            print("Oops!  That was no valid number.  Try again...")
    return x


print('Welcome to the Plaquette EDEN Model!')

print('Do you have a file with a model? \n0 -- you want to generate a new model \n1 -- you have a file')
file = bool(read_value([0, 1]))

print(
    'Which type of model? \n0 -- plaquette eden model \n1 -- euler-characteristic mediated plaquette eden model'
    '\n2 -- filled-in cubes plaquette eden model')
model = read_value([0, 1, 2])

print('Do you want a picture of your model? (with a large model it can take time)  \n0 -- no \n1 -- yes')
pic = bool(read_value([0, 1]))

if pic:
    print('Do you want Python or MAYA 3D model? (We wouldn\'t recommend Python for large models (more than 500 tiles)).'
          ' \n0 -- Python \n1 -- MAYA')
    maya = bool(read_value([0, 1]))

"""NO FILE CASE"""
if not file:
    print('How many tiles would you like in your model?')
    Time = read_value_int()

    print('How many models would you like to build?')
    num_models = read_value_int()

    for m in range(num_models):
        print("\nWORKING ON MODEL #" + str(m + 1))
        folder_name = create_directory(Time, m, model)

        Betti_1_total_vector, Per_2d, Per_3d, Betti_2_total_vector, Eden, Process, Created_holes, Holes, Barcode, \
            Vertices, Process_ripser, Inner_perimeter_2d, Voids, Per_2d_in, Per_3d_in, Inner_perimeter_3d, \
            Per_2d_in_array_, Per_3d_in_array_ , P2_, P3_ = grow_eden(Time, model)

        """BETTI NUMBERS AND PERIMETERS"""
        plot_b_per(Betti_1_total_vector, Betti_2_total_vector, Per_2d, Per_3d, Time, int(Time/3), folder_name, model)

        """FREQUENCIES B1"""
        freq, changes = return_frequencies_1(Betti_1_total_vector, Time)
        if model == 1:
            draw_frequencies_1_eu(freq, changes, folder_name)
        else:
            draw_frequencies_1(freq, folder_name)

        """FREQUENCIES B2"""
        if model != 1:
            freq, changes = return_frequencies_2(Betti_2_total_vector, Time)
            draw_frequencies_2(freq, folder_name)

        """LOCAL GEOMETRY"""
        if model != 1:
            Tricube, Tricube_f, Tetracube, Tetracube_f = num_holes(Created_holes, Holes)
            draw_tri_tetra(Tricube, Tricube_f, Tetracube, Tetracube_f, folder_name)
            draw_diagram_holes(Created_holes, Holes, folder_name)

        """BARCODE FOR B2"""
        if model != 1:
            print("Plotting Betti_2 Barcode...")
            if model == 2:
                Barcode = [a for a in Barcode if a[1][1] - a[1][0] != float('inf')]
            draw_barcode_gudhi(Barcode, folder_name, 2)
        if model == 2:
            draw_pers_diagram([x[1] for x in Barcode], [x[1] for x in Barcode], Time, folder_name, Per_2d)

        """DISTANCE MATRIX FOR RIPSER"""
        create_dist_matrix(Time, Process_ripser, Vertices, folder_name)

        """BARCODE FOR B1"""
        # folder_barcode = '.'
        # Barcode_b1 = read_barcode_b1_from_file(folder_barcode)
        # Barcode_b1 = [a for a in Barcode_b1 if a[1][1] - a[1][0] != float('inf')]
        # draw_barcode_gudhi(Barcode_b1, folder_name, 1)

        if pic:
            if maya:
                f = open(folder_name + "/MAYA.txt", "w+")
                f.write("import maya.cmds as cmds \n"
                        "Eden = " + str(Process) + "\nt = len(Eden)"
                                                   "\nfor i in range(t):"
                                                   "\n\tcmds.polyCreateFacet(p = Eden[i])")
                f.close()
                print("We created txt file \"MAYA\" for you. Just copy paste its content to MAYA!")
            else:
                draw_eden(Eden, folder_name, Time)
                print("Python 3D model is created!")


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

