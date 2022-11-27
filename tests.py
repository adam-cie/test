from main_3 import MSDTransformer
import numpy as np
import pandas as pd
import pytest

def test_mytest():

    df = pd.read_csv("bus.csv", sep = ';', index_col = 0)

    objectives = ['max','max','min','max','min','min','min','max']

    TopsisTransformer = MSDTransformer()
    with pytest.raises(Exception):
        TopsisTransformer.transform()
