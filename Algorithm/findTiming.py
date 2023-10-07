import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    fileName = "time_taking_no_convex.csv"
    #fileName = "time_taking_with_convex.csv"
    Dataframe = pd.read_csv(fileName)
    #print(Dataframe)
    #print(Dataframe[""])
    print(Dataframe["main"].mean())
    print(Dataframe["cage_detector"].mean())
    print(Dataframe["main"].min(), Dataframe["main"].max())
    print(Dataframe["cage_detector"].min(), Dataframe["cage_detector"].max())
    ax = Dataframe.hist()
    print(ax)
    #plt.hist(ax)
    ax[1,0].plot()
    plt.show()