import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.offsetbox import AnchoredText


def linear_fit(X, y):
    """
    Fit correlation between independent variables.
    ----------------------------------------
    :param X: Independent variable, pandas Series.
    :param y: Another independent variable to predict, pandas Series.
    :return: X to be plotted and its corresponding prediction.
    """
    model = LinearRegression()
    plot_X = np.linspace(X.min(), X.max()).reshape(-1, 1)
    model.fit(X.values.reshape(-1, 1), y.values.reshape(-1, 1))
    plot_y = model.predict(plot_X)
    return plot_X, plot_y


def corr_plot(dataframe, fig_height=None, fig_width=None, **kwargs):
    """
    Plot correlation matrix of a dataframe, and save to local directory.
    ----------------------------------------
    :param dataframe: dataframe to be plot.
    :param fig_height: height of the figure to be plot, default=None.
    :param fig_width: width of the figure to be plot, default=None.
    :return: None
    """
    corr_matrix = dataframe.corr()
    n = len(corr_matrix)
    fig, axes = plt.subplots(nrows=n, ncols=n, sharex='col')
    if fig_height is not None:
        fig.set_figheight(fig_height)
    if fig_width is not None:
        fig.set_figwidth(fig_width)
    subplot = 1
    for col in corr_matrix.columns:
        for idx in corr_matrix.index:
            plt.subplot(n, n, subplot)
            ax = axes[corr_matrix.columns.get_loc(col)][corr_matrix.index.get_loc(idx)]
            if col != idx:
                plt.scatter(dataframe[idx], dataframe[col], **kwargs)
                fit_X, fit_y = linear_fit(dataframe[idx], dataframe[col])
                plt.plot(fit_X, fit_y, color='r')
                at = AnchoredText(round(corr_matrix[col][idx], 2),
                                  prop=dict(size=12), frameon=True,
                                  loc='upper left',
                                  )
                at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
                ax.add_artist(at)
            else:
                plt.hist(dataframe[col], bins=10, edgecolor='black')
            if subplot % n == 1:
                plt.ylabel(col)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            if subplot / n > n - 1:
                plt.xlabel(idx)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
            subplot += 1
    plt.tight_layout()
    plt.savefig('corr_plot.png', dpi=300)
    return None


if __name__ == '__main__':
    df = pd.read_csv('auto_clean.csv')
    df = df[['price', 'highway-mpg', 'curb-weight', 'horsepower', 'length', 'width']]
    corr_plot(df, 12, 18)
