{\rtf1\ansi\ansicpg1252\cocoartf2511
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 HelveticaNeue;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
{\*\listtable{\list\listtemplateid1\listhybrid{\listlevel\levelnfc0\levelnfcn0\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{decimal\}.}{\leveltext\leveltemplateid1\'02\'00.;}{\levelnumbers\'01;}\fi-360\li720\lin720 }{\listname ;}\listid1}
{\list\listtemplateid2\listhybrid{\listlevel\levelnfc0\levelnfcn0\leveljc0\leveljcn0\levelfollow0\levelstartat22\levelspace360\levelindent0{\*\levelmarker \{decimal\}.}{\leveltext\leveltemplateid101\'02\'00.;}{\levelnumbers\'01;}\fi-360\li720\lin720 }{\listname ;}\listid2}
{\list\listtemplateid3\listhybrid{\listlevel\levelnfc0\levelnfcn0\leveljc0\leveljcn0\levelfollow0\levelstartat23\levelspace360\levelindent0{\*\levelmarker \{decimal\}.}{\leveltext\leveltemplateid201\'02\'00.;}{\levelnumbers\'01;}\fi-360\li720\lin720 }{\listname ;}\listid3}}
{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}{\listoverride\listid2\listoverridecount0\ls2}{\listoverride\listid3\listoverridecount0\ls3}}
\paperw11900\paperh16840\margl1440\margr1440\vieww28600\viewh18000\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 Hi Erika!\
\
Here are some notes about my changes:\
\
\pard\pardeftab560\pardirnatural\partightenfactor0
\ls1\ilvl0
\f1 \cf0 {\listtext	1.	}Separated it in 4 files\
{\listtext	2.	}Diff -> diff (pycharm error: name of functions should be lowercase)\
{\listtext	3.	}Added some comments/suggestions\
{\listtext	4.	}Added some spaces (also pycharm error)\
{\listtext	5.	}Added default color to draw_square() as grey\
{\listtext	6.	}Added ls as a argument of draw_square() with default value 0.35\
{\listtext	7.	}Eden, Perimeter -> eden, perimeter \
{\listtext	8.	}Holes -> holes\
{\listtext	9.	}Created_Holes -> created_holes\
{\listtext	10.	}Vertices -> vertices\
{\listtext	11.	}Edges -> edges\
{\listtext	12.	}num_posible_components -> num_possible_components (I guess posible it\'92s a Spanish word, am I right?:) )\
{\listtext	13.	}start_eden_2D() - > start_eden_2d()\
{\listtext	14.	}if holes[num_hole] == [] -> if not holes[num_hole]\
{\listtext	15.	}Process - > process\
{\listtext	16.	}grow_eden_debuging -> grow_eden_debugging\
{\listtext	17.	}Added argument time to draw_polyomino() and now the name of a figure is 'eden_'+str(time)+'.svg'\
{\listtext	18.	}Same with draw_polyomino_holes()\
{\listtext	19.	}I think there is a typo in draw_barcode() and instead of \'91if barcode[x][1] == 0:\'92 there should be \'91if barcode[x][0] == 0:\'92\
{\listtext	20.	}Edited the barcode function (it didn\'92t work for me at all)\
{\listtext	21.	}Cut the part into small cycle:\
\pard\pardeftab560\slleading20\pardirnatural\partightenfactor0
\cf0 	for n in [n1, n2, n3, n4]:\
	    if n in eden:\
        		eden[n][1] = eden[n][1] + 1\
           if eden[n][0] == 1:\
                 nearest_n[0] = 1\
    else:\
        eden[n] = [0, 1, eden[tile_selected][2]]\
        perimeter = perimeter + [n]\
\pard\pardeftab560\pardirnatural\partightenfactor0
\ls2\ilvl0\cf0 	22. The next line look weird: function  increment_betti_2() return betti_2 and we initialise a return of it to betti_1\
\pard\pardeftab560\slleading20\pardirnatural\partightenfactor0
\cf0 		betti_1, total_holes, eden, barcode, holes, created_holes, tags = increment_betti_2(eden, tile_selected,\
                                                                                    nearest_n,\
                                                                                    nearest_n_tiles, barcode, i,\
                                                                                    holes, total_holes,\
                                                                                    created_holes, tags)\
\pard\pardeftab560\pardirnatural\partightenfactor0
\ls3\ilvl0\cf0 {\listtext	23.	}Everywhere where we create tuples, I did it on initialisation step\
}