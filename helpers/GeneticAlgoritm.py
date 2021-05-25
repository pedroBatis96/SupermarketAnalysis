import random
from multiprocessing import Process, Manager

import networkx as nx
import numpy as np
import pandas as pd
import json
import random
import collections

from helpers.GraphHelper import GraphHelper
from helpers.TheSimulator import TheSimulator
from tqdm import tqdm


def create_reference():
    df_p = pd.read_csv('data/products.csv', encoding='utf-8', usecols=['ID', 'Total Prateleiras'], index_col="ID")

    distribution = []
    for index, row in df_p.iterrows():
        for prateleira in range(0, row['Total Prateleiras']):
            distribution.append(index)
    return np.array(distribution, dtype=int)


class TheGenetic:
    reference = None
    graph = None
    simulator = None
    mode = 'SALES'
    chromosomes = []
    epoch = 1

    def __init__(self):
        self.reference = create_reference()
        self.graph = GraphHelper().create_supermarket()
        self.simulator = TheSimulator(self.graph)

        # cria a lista inicial de cromossomas
        self.chromosomes = []
        for i in range(0, 6):
            np.random.shuffle(self.reference)
            array_aux = self.reference.copy()
            self.chromosomes.append([array_aux, 0, []])

        self.chromosomes = np.array(self.chromosomes, dtype=tuple)
        self.start_trein()

    def start_trein(self):
        self.simulator.prepare_clients(200)

        for c, cro in enumerate(tqdm(self.chromosomes)):
            self.simulator.prepare_products(cro[0])
            sales, profits, total_access = self.simulator.begin_simulation()
            if self.mode == 'SALES':
                self.chromosomes[c][1] = sales
            else:
                self.chromosomes[c][1] = profits
            self.chromosomes[c][2] = total_access

        self.mutate()

    def mutate(self, alpha=0.3, top=20):
        worst_cromossomes = self.chromosomes[:, 1].argsort()[0:3]
        self.chromosomes = np.delete(self.chromosomes, worst_cromossomes, axis=0)

        for chrom in self.chromosomes:
            unmovable_shelves = (-chrom[2]).argsort()[:top]
            movable_shelves = [i for i in range(0, 248) if i not in unmovable_shelves]
            new_chromossome = chrom[0].copy()

            for p, product in enumerate(chrom[0]):
                if p in unmovable_shelves:
                    new_chromossome[p] = product
                else:
                    sigma = random.uniform(0, 1)
                    if sigma <= alpha:
                        twist_p = -1
                        while twist_p == p or twist_p == -1:
                            twist_p = random.choice(movable_shelves)
                        new_chromossome[p], new_chromossome[twist_p] = new_chromossome[twist_p], new_chromossome[p]