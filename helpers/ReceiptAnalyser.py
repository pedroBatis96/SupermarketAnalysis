import csv
import pandas as pd
import numpy as np
import time
from pathlib import Path


# para cada recibo de cada pasta, analisa e retira a informação relevante
def analyse_receipts(start_r, end_r,return_dict):
    ra = ReceiptAnalyser()
    p_size = len(ra.prod_df.index)
    for d in range(start_r, end_r):
        n_products = np.zeros(p_size, dtype=int)

        # start2 = time.time()
        receipt = pd.DataFrame(columns=['ogFile', 'nif', 'products_all', 'products', 'groups', 'total'])
        dir_str = "../receipts/{}/".format(d)
        print("\nreceipts_{}".format(d) + " started:")

        for path in Path(dir_str).rglob('*.txt'):
            information, n_products = ra.analyze_receipt(path, path.name, n_products)
            receipt = receipt.append(information, ignore_index=True)

        receipt.reindex()
        receipt.to_csv('data/receipt_{}.csv'.format(d), encoding="utf-8")

        return_dict["r{}".format(d)] = n_products
        print("\nreceipts_{}".format(d) + " ended")
        # print(time.time() - start2)


class ReceiptAnalyser:
    prod_dict = None
    proud_group_dict = None
    prod_df = None

    def __init__(self, prod_df=None):
        if prod_df is None:
            self._get_products_dicts()
        else:
            self.prod_df = prod_df

    # transforma os grupos e ids dos items em dicionarios
    def _get_products_dicts(self):
        self.prod_df = pd.read_csv('data/products.csv', encoding='utf-8', index_col=["Nome", "Preço"],
                                   usecols=["ID", "Nome", "Grupo", "Preço"], dtype={"Preço": float})

    # dissecação de cada recibo
    def analyze_receipt(self, filepath, filename, n_products):
        is_prod_zone = True
        information = {
            'ogFile': filename,
            'products_all': [],
            'products': [],
            'groups': [],
            'nif': [],
            'total': [],
        }

        with open(filepath, 'r', encoding='utf-8') as receipt:
            for i, line in enumerate(receipt):
                # linhas ignoráveis
                if i < 1 or (1 < i < 5):
                    continue
                # nif
                elif i == 1:
                    line = line.split(' ')
                    if len(line) < 3:
                        print('Este recibo tem dados em falta')
                    else:
                        information['nif'] = int(line[2].strip())
                # produtos
                elif i > 5 and is_prod_zone:
                    if line[1] == '>':
                        line = line[2:]
                        line = line.split(':')

                        # 0 product id, 1 group id
                        product = self.prod_df.loc[line[0].strip(), float(line[1].strip())]
                        product = product.to_numpy()

                        n_products[(product[0] - 1)] += 1
                        # append to respective array
                        information['products_all'].append(product[0])
                        if product[0] not in information['products']:
                            information['products'].append(product[0])
                        if product[1] not in information['groups']:
                            information['groups'].append(product[1])
                        continue
                    else:
                        is_prod_zone = False
                        continue
                else:
                    line = line.split(':')
                    information['total'] = float(line[1].strip().split(' ')[0])

        return information, n_products
