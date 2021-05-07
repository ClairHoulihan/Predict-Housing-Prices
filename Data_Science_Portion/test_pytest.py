#!/usr/bin/env python
import predicting
import pandas as pd
import pytest

@pytest.mark.parametrize("cols", [["Street", "Neighborhood", "OverallQual", "OverallCond", "RoofStyle"],
["3SsnPorch", "PoolArea", "YrSold"], []])

def test__init__(cols):
    testing = predicting.Housing_Model(columns=cols)
    cols.append("SalePrice")
    assert list(testing.training_set.columns) == cols
