from main_3 import MSDTransformer
import numpy as np
import pandas as pd
import pytest

class TestParameters:
    df = pd.read_csv("bus.csv", sep = ';', index_col = 0)
    objectives = ['max','max','min','max','min','min','min','max']

    def test_single_objective(self):
        for i in ['c', 'g', 'min', 'max', 'gain', 'cost']:
            buses = MSDTransformer()
            buses.fit(self.df, objectives = i)
            buses.transform()
            assert 1==1