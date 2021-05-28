import numpy as np
import networkx as nx
from os import path

'''
2 = entrada
3 = saida
1 = prateleira
-1 = prateleira inacessivel
'''


class GraphHelper:
    walk_tiles = []
    supermarket_type = []
    supermarket_number = []
    connections = {}

    def __init__(self):
        self.walk_tiles = [
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            48, 64, 65, 66, 68,
            71, 87, 88, 89, 91,
            94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
            117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
            140, 156, 157, 160,
            163, 179, 180, 183,
            163, 179, 180, 183,
            186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206,
            209, 225, 226, 227, 228, 229,
            232, 248, 249, 252,
            255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 275,
            278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 298,
            301, 317, 318, 319, 320, 321,
            324, 340, 341, 342, 343, 344,
            347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 367,
            370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 390,
            393, 409, 410, 413,
            416, 432, 433, 436,
            439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459
        ]

    # marca a matriz como a legenda no inicio, e atribui um número
    def mark_territory(self) -> int:
        n = 0
        for i in range(0, 21):
            for aux in range(0, 23):
                n += 1
                if n == 115:
                    shelf_type = 3
                elif n == 414:
                    shelf_type = 2
                elif n in self.walk_tiles:
                    shelf_type = 0
                elif n in [1, 23, 461, 483]:
                    shelf_type = -1
                else:
                    shelf_type = 1

                self.supermarket_number[aux, i] = n
                self.supermarket_type[aux, i] = shelf_type

    # cria um dicionário com as conexões para cada um
    def create_connections(self):
        for i in range(0, 21):
            for aux in range(0, 23):
                connect_aux = []
                if self.supermarket_type[aux, i] == 0:
                    connect_aux.append(self.supermarket_number[aux - 1, i])
                    connect_aux.append(self.supermarket_number[aux + 1, i])
                    connect_aux.append(self.supermarket_number[aux, i - 1])
                    connect_aux.append(self.supermarket_number[aux, i + 1])
                if self.supermarket_number[aux, i] in [414, 115]:
                    connect_aux.append(self.supermarket_number[aux - 1, i])
                    connect_aux.append(self.supermarket_number[aux, i - 1])
                    connect_aux.append(self.supermarket_number[aux, i + 1])

                self.connections[self.supermarket_number[aux, i]] = connect_aux

    def create_supermarket(self):
        if not path.exists("data/extras/supermarket.gpickle"):
            self.supermarket_number = np.zeros([23, 21], dtype=int)
            self.supermarket_type = np.zeros([23, 21], dtype=int)
            self.mark_territory()
            self.create_connections()

            graph = nx.DiGraph(self.connections)

            nx.write_gpickle(graph, "data/extras/supermarket.gpickle")
        else:
            graph = nx.read_gpickle("data/extras/supermarket.gpickle")
        return graph


#
def create_matrix():
    pass
