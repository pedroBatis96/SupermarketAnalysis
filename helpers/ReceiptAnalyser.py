import csv
import pandas as pd


class ReceiptAnalyser:
    prod_dict = None
    proud_group_dict = None

    def __init__(self):
        self._get_products_dicts()

    # transforma os grupos e ids dos items em dicionarios
    def _get_products_dicts(self):
        product_dataframe = pd.read_csv('data/products.csv', encoding='utf-8', index_col=0)
        self.prod_dict = dict(zip(product_dataframe.Nome, product_dataframe.index))
        self.proud_group_dict = dict(zip(product_dataframe.Nome, product_dataframe.Grupo))

    def analyze_receipt(self, r):
        i = 1

        information = {
            "total": 0,
            "nif": 0,
            "produtos": [],
            "grupos": []
        }
        is_prod_zone = True
        for line in iter(r.splitlines()):
            # nao interessam
            if i < 2:
                i += 1
                continue
            # tratamento do nif
            if i == 2:
                split_l = line.split(' ')
                if len(split_l) < 3:
                    print('Este recibo tem dados em falta')
                else:
                    information['nif'] = int(split_l[2])
                i += 1
                continue
            # tratamento do nif
            if i > 5 and is_prod_zone:
                # tratamento dos produtos, acrescentando o seu id e id de grupo nos correspondentes
                if line[1] == '>':
                    line = line[2:]
                    line = line.split(':')
                    product = line[0].strip()
                    product_id = self.prod_dict[product]
                    product_group_id = self.proud_group_dict[product]

                    if product_id not in information['produtos']:
                        information['produtos'].append(product_id)
                    if product_group_id not in information['grupos']:
                        information['grupos'].append(product_group_id)
                # simboliza o final da zona dos produtos
                elif line[1] == '-':
                    is_prod_zone = False
                    i += 1
                    continue
            # verificar o total do recibo
            if i > 5 and not is_prod_zone:
                line = line.split(':')
                information['total'] = line[1].strip().split(' ')[0]
            i += 1
        print(information['produtos'])
        print(information['grupos'])
