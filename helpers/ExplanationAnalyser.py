import pandas as pd
from pathlib import Path
import re


# para cada explicação de cada pasta, analisa e retira a informação relevante
def analyse_explanations(start_r, end_r):
    ra = ExplanationAnalyser()
    for d in range(start_r, end_r):

        # start2 = time.time()
        receipt = pd.DataFrame(columns=['ogFile', 'products', 'groups', 'stamina'])
        dir_str = "../explanations/{}/".format(d)
        print("\nexplanation_{}".format(d) + " started:")

        for path in Path(dir_str).rglob('*.txt'):
            information = ra.analyze_receipt(path, path.name)
            receipt = receipt.append(information, ignore_index=True)

        receipt.reindex()
        receipt.to_csv('data/explanations/explanation_{}.csv'.format(d), encoding="utf-8")

        print("\nexplanation_{}".format(d) + " ended")
        # print(time.time() - start2)


class ExplanationAnalyser:
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
        self.prod_df = pd.read_csv('data/products.csv', encoding='utf-8', index_col=["Nome"],
                                   usecols=["ID", "Nome", "Grupo"])

    # dissecação de cada recibo
    def analyze_receipt(self, filepath, filename):
        information = {
            'ogFile': filename,
            'products': [],
            'groups': [],
            'stamina': [],
        }

        with open(filepath, 'r', encoding='utf-8') as receipt:
            lines = receipt.readlines()
            for i, line in enumerate(lines):
                if i > 0:
                    if line[1] == '-':
                        line = line[2:]
                        line = line.strip()

                        # 0 product id, 1 group id
                        product = self.prod_df.loc[line]
                        product = product.to_numpy()

                        # append to respective array
                        if product[0] not in information['products']:
                            information['products'].append(product[0])
                        if product[1] not in information['groups']:
                            information['groups'].append(product[1])
                        continue
                    else:
                        break
            # verifica a stamina do cliente
            #caso tenha ficado cansado é na penultima, caso nao é na ultima
            if "Walked" in lines[-2]:
                stamina_values = re.findall(r"[-+]?\d*\.\d+|\d+", lines[-2])
            elif "Walked" in lines[-1]:
                stamina_values = re.findall(r"[-+]?\d*\.\d+|\d+", lines[-1])
            information['stamina'] = float(stamina_values[0]) + float(stamina_values[1])

        return information
