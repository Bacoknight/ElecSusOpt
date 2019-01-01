"""
A small module used for reading the results from a chocolate database. Useful if the program crashes before the end as it allows us to find
the values it obtained before crashing.
"""

import pandas as pd
import chocolate as choco
import numpy as np
from elecsus import elecsus_methods as elecsus
import two_filters as filters

def GetBest5Results(databaseConnection, toPrint = True):
    """
    This function prints the best 5 ElecSus parameters for a given database, and their respective figures of merit.
    It also returns the best parameters found.
    """
    assert type(databaseConnection) == type(choco.SQLiteConnection("sqlite:///dummy.db")), "Input databaseConnection is not the correct type."

    resultsTable = databaseConnection.results_as_dataframe()
    results = resultsTable.sort_values(by = "_loss", ascending = False)
    if toPrint:
        print("Top 5 results:")
        print(results.iloc[0:5])

    return results.to_dict('records')[0]

def ConfirmTwoFilters(databaseConnection):
    """
    This function ensures that the best figure of merit as reported by the database is reproducible, by reproducing it.
    """

    bestDict = GetBest5Results(databaseConnection, False)
    bestFoM = bestDict.pop("_loss", "None found")

    print("Best figure of merit found: " + str(bestFoM))
    
    realFoM = filters.TwoFilters(**bestDict)

    print("Figure of merit as determined by ElecSus: " + str(realFoM))

    return

if __name__ == "__main__":
    ConfirmTwoFilters(choco.SQLiteConnection("sqlite:///two_filters_db.db"))
