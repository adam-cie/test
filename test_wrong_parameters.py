from main_3 import MSDTransformer
import numpy as np
import pandas as pd
import pytest

class TestParameters:
    df = pd.read_csv("bus.csv", sep = ';', index_col = 0)
    objectives = ['max','max','min','max','min','min','min','max']

    def test_copy(self):
        buses = MSDTransformer()
        buses.fit(self.df, objectives=self.objectives)
        a = buses.transform()
        #assert self.df.equals(buses.data_)