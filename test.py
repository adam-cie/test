from main_3 import MSD_Transformer
import numpy as np
import pandas as pd

df = pd.read_csv("bus.csv", sep = ';')
alternatives = df['BusId'].to_numpy()
df = df.drop(columns=['BusId'])
criteria = df.columns.to_numpy()
data = df.to_numpy(dtype=float)
objectives = np.array(('max','max','min','max','min','min','min','max'))

buses = MSD_Transformer(data, criteria, alternatives, None, objectives, None, 'I')
buses.transform()
assert (buses.ranked_alternatives == np.array(['b24', 'b26', 'b07', 'b16', 'b18', 'b25', 'b04', 'b01', 'b28', 'b09', 'b02', 'b13', 'b11', 'b32', 'b21', 'b12', 'b27', 'b17', 'b06', 'b29', 'b20', 'b14', 'b23', 'b19', 'b03', 'b30', 'b08', 'b22', 'b15', 'b10', 'b31', 'b05'])).all(), "wrong ranking for bus database"