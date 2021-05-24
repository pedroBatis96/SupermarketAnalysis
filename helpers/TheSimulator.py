import random

import networkx as nx
import numpy as np
import pandas as pd
import json

from helpers.StatisticsHelper import get_normal_stamina_distribuition


class TheSimulator:
    graph = None
    clients = None
    walk_tiles = [
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
        48, 64, 65, 66, 68,
        71, 87, 88, 89, 91,
        94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
        116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136,
        137,
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

    none_tiles = [414, 115, 1, 23, 461, 483]
    product_distribution = {}


    # print(nx.dijkstra_path(self.graph, 414, 3))

    def __init__(self, graph):
        self.df_p = pd.read_csv('data/products.csv', usecols=['ID', 'Preço', 'Margem Lucro'], encoding='utf-8', index_col="ID")
        self.df_p['Lucro'] = self.df_p['Preço'] * (self.df_p['Margem Lucro'] / 100)

        # grafico e distribuição de produtos
        self.graph = graph
        self.distribute_products()

        # gerar clientes
        self.clients_generator(1000)
        self.begin_simulation()

        return

    # gera os clientes, começando por fazer a wishlist e a stamina
    def clients_generator(self, n):
        wishlist = self.get_wishlist_list(n)
        stamina_samples = self.get_stamina_samples(n)
        self.clients = self.client_creator(wishlist, stamina_samples)

    # cria os clientes
    def client_creator(self, wishlist, stamina_samples):
        clients = np.empty(len(wishlist), dtype=tuple)

        for i, item in enumerate(wishlist):
            clients[i] = (item, int(stamina_samples[i]))
        return clients

    # cria uma lista de wishlists
    def get_wishlist_list(self, n):
        frames = []
        for i in range(0, 50):
            df = pd.read_csv('data/explanations/explanation_{}.csv'.format(i), usecols=['products'],
                             dtype={"stamina": float})
            frames.append(df)

        result = pd.concat(frames)
        wish_list = list(result['products'].apply(json.loads).to_numpy())
        return random.sample(wish_list, n)

    # cria uma lista de samples
    def get_stamina_samples(self, n):
        stamina_dist = get_normal_stamina_distribuition()
        stamina_samples = stamina_dist.sample(n)
        return stamina_samples

    # distribui produtos pelo grafo
    def distribute_products(self):
        df_p = pd.read_csv('data/products.csv', encoding='utf-8', usecols=['ID', 'Total Prateleiras'], index_col="ID")

        p = 1
        nodes = self.graph.nodes
        for n in nodes:
            if n not in self.walk_tiles and n not in self.none_tiles:
                if df_p.at[p, 'Total Prateleiras'] > 0:
                    df_p.at[p, 'Total Prateleiras'] -= 1

                    if p not in self.product_distribution.keys():
                        self.product_distribution[p] = []

                    self.product_distribution[p].append(n)
                    self.graph.nodes[n]['p'] = p
                else:
                    p += 1
                    df_p.at[p, 'Total Prateleiras'] -= 1
                    self.product_distribution[p] = []
                    self.product_distribution[p].append(n)
                    self.graph.nodes[n]['p'] = p
            else:
                continue

    # na wishlist, reparei, que ele vai de baixo para cima
    def begin_simulation(self):
        for client in self.clients:
            total_client_profit = 0
            total_client_value = 0

            stamina = client[1]
            wish_list = client[0]
            bought_products = []

            current_position = 414
            while len(wish_list) > 0 and stamina > 0:
                next_p = wish_list[-1]

                # verifica a posição mais proxima do produto desejado
                shortest_path = None
                for node in self.product_distribution[next_p]:
                    path = nx.dijkstra_path(self.graph, current_position, node)
                    if not shortest_path or len(path) < len(shortest_path):
                        shortest_path = path

                # percorrer path
                for ti, tile in enumerate(path[0:-1]):
                    if ti == 0 and tile != 414:
                        continue

                    stamina -= 1

                    print("walked to {}".format(tile))
                    current_position = tile

                    # test boundaries
                    for shelf in self.graph.neighbors(current_position):
                        if shelf in self.walk_tiles or shelf in self.none_tiles:
                            continue

                        shelf_product = self.graph.nodes[shelf]['p']
                        if shelf_product in wish_list:
                            print("product in wish_list , bought {}".format(shelf_product))
                            bought_products.append(shelf_product)
                            wish_list.remove(shelf_product)

                            total_client_profit += self.df_p.at[shelf_product, 'Lucro']
                            total_client_value += self.df_p.at[shelf_product, 'Preço']

                print(total_client_profit,total_client_value)
