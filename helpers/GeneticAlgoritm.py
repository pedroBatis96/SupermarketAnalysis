import numpy as np
import pandas as pd
import random

from sklearn.neighbors import KernelDensity

from helpers.GraphHelper import GraphHelper
from helpers.TheSimulator import TheSimulator
from tqdm import tqdm

from scipy.stats import norm
from collections import Counter


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
    mode = 'sales'
    chromosomes = []
    epoch = 1
    the_king = [[], 0, []]

    def __init__(self, mode='sales'):
        self.mode = mode
        self.reference = create_reference()
        self.graph = GraphHelper().create_supermarket()
        self.simulator = TheSimulator(self.graph)

        # cria a lista inicial de cromossomas
        self.chromosomes = []
        for i in range(0, 10):
            np.random.shuffle(self.reference)
            array_aux = self.reference.copy()
            self.chromosomes.append([array_aux, 0, []])

        self.chromosomes = np.array(self.chromosomes, dtype=tuple)

    def start_train(self):
        alpha = 0.25
        top = 150
        for e in range(0, 10):
            print(f"Start Epoch {self.epoch}")
            # prepara n clientes que serão a população usada na função de aptidao
            self.simulator.prepare_clients(15000)

            # executa a função de aptidao
            for c, cro in enumerate(tqdm(self.chromosomes)):
                self.simulator.prepare_products(cro[0])
                sales, profits, total_access = self.simulator.begin_simulation()
                if self.mode == 'sales':
                    self.chromosomes[c][1] = sales
                else:
                    self.chromosomes[c][1] = profits
                self.chromosomes[c][2] = total_access

            save_dict = self._chromossome_to_dict()
            pd.DataFrame(save_dict).to_csv(
                f"data/genetics/{self.mode}/genetic_epoc_{self.epoch}.csv", encoding='utf-8')

            # se for a ultima epoca nao faz mutação
            if e != 9:
                self.mutate(alpha, top)
                alpha += 0.05
                top -= 10

            self.epoch += 1

        # guarda a informação num dicionario para record, incluindo a do king
        save_dict = self._chromossome_to_dict()
        save_dict['distribution'].append(self.the_king[0].tolist())
        save_dict['total'].append(self.the_king[1])
        save_dict['sale_by_shelf'].append(self.the_king[2].tolist())

        pd.DataFrame(save_dict).to_csv(f"data/genetics/{self.mode}/geneticresults.csv", encoding='utf-8')

    def _chromossome_to_dict(self):
        save_dict = {'distribution': [], 'total': [], 'sale_by_shelf': []}
        for c in self.chromosomes:
            save_dict['distribution'].append(c[0].tolist())
            save_dict['total'].append(c[1])
            save_dict['sale_by_shelf'].append(c[2].tolist())
        return save_dict

    def mutate(self, alpha=0.3, top=20):
        # apaga os n piores cromossomas (seleção por elitismo)
        worst_cromossomes = self.chromosomes[:, 1].argsort()[0:5]
        self.chromosomes = np.delete(self.chromosomes, worst_cromossomes, axis=0)
        del worst_cromossomes

        # vai buscar o melhor cromossoma e substitui pelo rei se for melhor
        top_cromossome = self.chromosomes[(-self.chromosomes[:, 1]).argsort()[:1]][0].copy()
        if top_cromossome[1] > self.the_king[1]:
            self.the_king = top_cromossome.copy()

        print(top_cromossome[1])
        print(self.chromosomes[:, 1])
        del top_cromossome

        new_chromosomes = []
        for chrom in self.chromosomes:
            new_chromossome = mutate_chromossome(chrom, top)

            if Counter(new_chromossome) != Counter(self.reference):
                raise Exception("Wrong Counter")

            new_chromosomes.append([np.array(new_chromossome), 0, []])

        new_chromosomes = np.array(new_chromosomes, dtype=tuple)
        self.chromosomes = np.concatenate((self.chromosomes, new_chromosomes), axis=0)
        del new_chromosomes


def mutate_chromossome(chromossome, top):
    # verifica quais foram as top prateleiras vendidas e adiciona as às prateleiras nao moviveis
    movable_shelves = list((chromossome[2].copy()).argsort()[:top])

    chrom_aux = []
    for mov in movable_shelves:
        chrom_aux.append(chromossome[0][mov])
    random.shuffle(chrom_aux)

    for i, mov in enumerate(movable_shelves):
        chromossome[0][mov] = chrom_aux[i]

    return list(chromossome[0])


def get_kernel_density(X):
    mu, std = norm.fit(X)
    return np.vectorize(norm(mu, std).cdf)(X)


def old_mode(chrom, top, alpha):
    # verifica quais foram as top prateleiras vendidas e adiciona as às prateleiras nao moviveis
    unmovable_shelves = list((-chrom[2].copy()).argsort()[:top])
    movable_shelves = [i for i in range(0, 248) if i not in unmovable_shelves]
    new_chromossome = list(chrom[0].copy())

    # objeto que ajuda a calcular o quao baixa é a probabilidade
    shelf_prob = get_kernel_density(list(chrom[2].reshape(1, -1)[0]))

    # começa a recriar uma recriação de um cromossoma de topo, em que muda as piores prateleiras entre elas
    for p, product in enumerate(chrom[0]):
        if p in unmovable_shelves:
            new_chromossome[p] = product
        else:
            sigma = random.uniform(0, 1)

            # probabilidade de fazer uma mutação , verifica se é maior que alpha e se a prob de mudar é baixa
            if alpha >= sigma or alpha > shelf_prob[p]:
                twist_p = -1

                # Escolher um ao acaso para trocar
                while twist_p == p or twist_p == -1:
                    twist_p = random.choice(movable_shelves)

                new_chromossome[p], new_chromossome[twist_p] = new_chromossome[twist_p], new_chromossome[p]
