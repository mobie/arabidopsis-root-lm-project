import pandas as pd
table_path = './data/arabidopsis-root/tables/lm-cells/default.tsv'

tab = pd.read_csv(table_path, sep='\t')
print(len(tab))
tab = tab[tab['label_id'] != 0]
print(len(tab))

tab.to_csv(table_path, sep='\t', index=False)
