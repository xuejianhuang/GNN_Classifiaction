import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import config


def viewConfusin(confusin_df):

    lows=confusin_df.shape[0]
    for i in range(lows):
        confusin_df.iloc[i] = confusin_df.iloc[i] / confusin_df.iloc[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusin_df.to_numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + config.label, rotation=0)
    ax.set_yticklabels([''] + config.label)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig('confusion.png',dpi=600)
    #plt.show()

if __name__ == '__main__':
    confusin_df=pd.read_csv('./results/gat_3_3_confusin.csv')
    viewConfusin(confusin_df)
    


