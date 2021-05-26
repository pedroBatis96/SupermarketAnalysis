import random
from multiprocessing import Process, Manager

import networkx as nx
import numpy as np
import pandas as pd
import json

from helpers.DataMiningHelper import calc_random_pick_up_prob
from helpers.StatisticsHelper import get_normal_stamina_distribuition


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARN = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class TheSimulator:
    graph = None
    clients = None
    walk_tiles = np.array([
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
        439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459,
        414, 115, 1, 23, 461, 483
    ])
    shelf_tiles = np.setdiff1d(np.arange(1, 483), walk_tiles).tolist()
    # none_tiles = np.array([])
    product_distribution = {}
    df_p = None

    def __init__(self, graph):
        self.initialize_products_df()

        # grafo
        self.graph = graph
        return

    def initialize_products_df(self):
        # cria um dataframe para os produtos
        self.df_p = pd.read_csv('data/products.csv',
                                usecols=['ID', 'Nome', 'Preço', 'Margem Lucro', 'Total Prateleiras'], encoding='utf-8',
                                index_col="ID")
        self.df_p['Lucro'] = self.df_p['Preço'] * (self.df_p['Margem Lucro'] / 100)

        # faz merge com a probabilidade de ser apanhado ao acaso
        df_p_pickup = calc_random_pick_up_prob()
        self.df_p = pd.merge(self.df_p, df_p_pickup, on='ID')

    # region client generator
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

    # endregion  client generator

    # region produtos
    # distribui produtos pelo grafo
    def distribute_products(self, distribution=None):
        if distribution is None:
            distribution = []
            for index, row in self.df_p.iterrows():
                for prateleira in range(0, row['Total Prateleiras']):
                    distribution.append(index)
            distribution = np.array(distribution, dtype=int)

        i = 0
        self.product_distribution = {}
        nodes = self.graph.nodes
        for n in nodes:
            if n not in self.walk_tiles:
                p = distribution[i]
                if p not in self.product_distribution.keys():
                    self.product_distribution[p] = []

                self.product_distribution[p].append(n)
                self.graph.nodes[n]['p'] = p
                i += 1
            else:
                continue

    # endregion produtos

    def prepare_simulation(self, client_n, products=None):
        self.distribute_products(products)
        self.clients_generator(client_n)

    def prepare_clients(self, client_n):
        self.clients = None
        self.clients_generator(client_n)

    def prepare_products(self, products):
        self.distribute_products(products)

    def begin_simulation(self, multi_process=True):
        manager = Manager()
        return_dict = manager.dict()

        if not multi_process:
            total_profit, total_sales = self.execute_simulation(self.product_distribution, self.clients)

        else:
            threads = []
            if len(self.clients) % 2 == 0:
                clients_split = np.split(self.clients, 8)
            else:
                clients_split = np.split(self.clients, 3)

            for c, split in enumerate(clients_split):
                thread = Process(target=self.execute_simulation,
                                 args=(self.product_distribution, split, c, return_dict))
                threads.append(thread)
                thread.start()

            # for t in threads:
            #    t.start()

            for t in threads:
                t.join()

            total_profit = 0
            total_sales = 0
            total_access = np.zeros(248, dtype=int)
            for k in return_dict.keys():
                total_profit += return_dict[k]['profit']
                total_sales += return_dict[k]['value']
                total_access += return_dict[k]['total_access']

            for t in threads:
                t.close()

        self.product_distribution = {}
        manager.shutdown()
        return total_profit, total_sales, total_access
        # print(return_dict)

    # na wishlist, reparei, que ele vai de baixo para cima
    def execute_simulation(self, product_distribution, clients, split_id=None, return_dict={}):
        total_profit = 0
        total_sales = 0
        total_access = np.zeros(248, dtype=int)
        # start all clients simulation
        for client in clients:
            total_client_profit = 0
            total_client_value = 0

            stamina = client[1]
            wish_list = client[0]

            current_position = 414

            # start client simulation
            while len(wish_list) > 0 and stamina > 0:
                next_p = wish_list[-1]

                # verifica a posição mais proxima do produto desejado
                shortest_path = None
                for node in product_distribution[next_p]:
                    path = nx.dijkstra_path(self.graph, current_position, node)
                    if not shortest_path or len(path) < len(shortest_path):
                        shortest_path = path

                # percorrer path
                for ti, tile in enumerate(path[0:-1]):
                    if ti == 0 and tile != 414:
                        continue

                    stamina -= 1

                    # print(f"{bcolors.ENDC}walked to {tile}")
                    current_position = tile

                    # test boundaries
                    for shelf in self.graph.neighbors(current_position):
                        if shelf in self.walk_tiles:
                            continue

                        shelf_product = self.graph.nodes[shelf]['p']
                        # caso ainda esteja na wishlist, ir buscar o produto
                        if shelf_product in wish_list:
                            # print(f"{bcolors.OKGREEN}Bought {self.df_p.at[shelf_product, 'Nome']}, product in WISHLIST")
                            wish_list.remove(shelf_product)

                            total_client_profit += self.df_p.at[shelf_product, 'Lucro']
                            total_client_value += self.df_p.at[shelf_product, 'Preço']
                            total_access[self.shelf_tiles.index(shelf)] += 1

                        # verificar a probabilidade de comprar sem estar na wishlisht
                        else:
                            sigma = random.uniform(0, 1)
                            if sigma < self.df_p.at[shelf_product, 'ProbPickUp']:
                                # print(f"{bcolors.WARN}Bought {self.df_p.at[shelf_product, 'Nome']}, product NOT IN WISHLIST")

                                total_client_profit += self.df_p.at[shelf_product, 'Lucro']
                                total_client_value += self.df_p.at[shelf_product, 'Preço']
                                total_access[self.shelf_tiles.index(shelf)] += 1

            # end client simulation
            # if len(wish_list) == 0:
            # print(f"{bcolors.OKGREEN}Bought Everything")
            # pass
            # if stamina <= 0:
            # print(f"{bcolors.FAIL}Ran out of stamina")
            # pass

            total_profit += total_client_profit
            total_sales += total_client_value

        if split_id:
            return_dict[split_id] = {'profit': total_profit, 'value': total_sales, 'total_access': total_access}
        else:
            return total_profit, total_sales, total_access
