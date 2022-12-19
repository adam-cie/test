from main_3 import MSDTransformer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv("bus.csv", sep = ';', index_col = 0)
#alternatives = df['BusId'].to_numpy()
#df = df.drop(columns=['BusId'])
#criteria = df.columns.to_numpy()
#data = df.to_numpy(dtype=float)
#normal:
objectives = ['max','max','min','max','min','min','min','max']

#for testing objectives
#objectives = ['max','max','min','max','min','min','min']

#def foo():
#    pass

#normal:
buses = MSDTransformer('I')

#for testing objectives
#buses = MSDTransformer(df, None, "objectives", None, 'I')
#for testing weights
#buses = MSDTransformer(df, [1], objectives, None, 'I')
#for testing expert_range
#buses = MSDTransformer(df, None, objectives, [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]], 'I')
#buses = MSDTransformer(df, None, objectives, [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]], 'I')
#buses = MSDTransformer(df, None, objectives, [['a',1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]], 'I')

buses.fit(df, None, objectives, None)
buses.transform()
#assert (buses.ranked_alternatives == np.array(['b24', 'b26', 'b07', 'b16', 'b18', 'b25', 'b04', 'b01', 'b28', 'b09', 'b02', 'b13', 'b11', 'b32', 'b21', 'b12', 'b27', 'b17', 'b06', 'b29', 'b20', 'b14', 'b23', 'b19', 'b03', 'b30', 'b08', 'b22', 'b15', 'b10', 'b31', 'b05'])).all(), "wrong ranking for bus database"
print(buses.ranked_alternatives)
#print(buses.topsis_val)
print(buses.data)

buses.plot()


