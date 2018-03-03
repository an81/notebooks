import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np

import plotly.plotly 
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
init_notebook_mode(connected=True)

from sklearn.cluster import KMeans, SpectralClustering
from scipy.spatial.distance import cdist
from sklearn import metrics

colors = ['#1b5eb7', '#1bb7a4', '#1bb751', '#88b71b', '#b7a21b', 
              '#b7661b', '#b7371b', '#b71b5c', '#9c1bb7', '#302333',
              '#9879ad', '#7a45f7', '#44d9f7', '#439af7', '#f74343',
              '#775303', '#757703', '#f3f70e', '#f7860d', '#e0af70']


def get_series(df, *args):
    """ 
    Returns series of prices by default.
    
    Parameterers:
    -------------
    df : dataframe with raw data load from jason file

    *Args:
    volume : returns series of volumes,
    market_cap : returns series of market capitalisation by available supply,
    """

    dates = np.array([datetime.fromtimestamp(df["volume_usd"].iloc[i][0]/1000) for i in range(len(df.index))])
    if "volume" in args: 
        values = np.array([df["volume_usd"].iloc[i][1] for i in range(len(df.index))])
    elif "market_cap" in args: 
        values = np.array([df["market_cap_by_available_supply"].iloc[i][1] for i in range(len(df.index))])
    else: 
        values= np.array([df["price_usd"].iloc[i][1] for i in range(len(df.index))])
    ts = pd.Series(values, index = dates)
    return ts

def get_dataframe(df):
    """
    Returns dataframe with price in dolars, traded volume and market capitalization.
    """

    dates = np.array([datetime.fromtimestamp(df["volume_usd"].iloc[i][0]/1000) for i in range(len(df.index))])
    price = np.array([df["price_usd"].iloc[i][1] for i in range(len(df.index))])
    volume = np.array([df["volume_usd"].iloc[i][1] for i in range(len(df.index))])
    market_cap = np.array([df["market_cap_by_available_supply"].iloc[i][1] for i in range(len(df.index))])
    res = pd.DataFrame(data = {"price_usd": price, "volume": volume, "market_cap": market_cap}, index = dates)
    return res

def get_diffs(ts, *args):
    """
    Return logarithmic differences by default.

    Parameters:
    -----------
    ts : pd.Series,
        Timeseries with values for which the differences are computed.

    *Args: 
    abs_diffs :  computes absolute differences, 
    relative_diffs : computes relative differences 
    """

    if "abs_diffs" in args:
        diffs = ts.diff(1)
    elif "relative_diffs" in args:
        diffs = ts.pct_change(1)       
    else:   
        diffs =(np.log(ts)-np.log(ts.shift(1)))
    return diffs


def get_vols(ts, window=30):
    logdiffs = get_diffs(ts)
    monthly_vol = pd.rolling_var(logdiffs, window)
    return monthly_vol

def plot_series(ts, plot_title, lags, *args):
    if "volume" in args:
        labels = "Volume in USD" 
    elif "market_cap" in args:
        labels = "Market Capitalisation in USD" 
    else:
        labels = "Price in USD"

    fig = {'data':
            [{'x': ts.index, 'y': ts.values,
            'mode': 'lines', 'name': labels, },] +  
            [{'x': ts.index, 
            'y': ts.rolling(window=lags[i],center=False).mean(),
            'mode': 'lines', 'line': {'color': colors[i+1]},      
            'name': 'Rolling Mean ' + str(lags[i]) + ' days',} for i in range(len(lags))],        
        'layout': 
        {'title': plot_title, 'yaxis': {'title': 'USD'} ,}}
    
    iplot(fig)
   # if "mean" in args: 
   #     for i in lags: 
   #          ts.rolling(window=i,center=False).mean().plot(figsize=[15,6])
   # if "vol" in args: 
   #     for i in lags: 
   #         ts.rolling(window=i,center=False).var().plot(figsize=[15,6])i
    
def plot_diffs(ts, plot_title, **kwargs):
    logdiffs = get_diffs(ts)
    fig = {'data':
            [{'x': logdiffs.index, 'y': logdiffs.values,
            'mode': 'lines', 'name': 'logarithmic differences'}, 
             {'x': logdiffs.index, 'y': logdiffs.rolling(window=7,center=False).var(),
             'mode': 'lines', 'line': {'color': colors[3]},
              'name': 'Rolling Variance 7 days'}],
            'layout': 
           {'title': plot_title, 'yaxis': {'title': 'USD'},}}
    iplot(fig)

def plot_histogram(ts, plot_title, **kwargs): 
    logdiffs = get_diffs(ts)
    fig = {'data':
            [{'x': logdiffs.values,
              'type': 'histogram', 
            'histnorm': 'count', 'name': "distribution of logarithmic differences", },],
          'layout': 
          {'title': plot_title, 
            'xaxis': {'title': 'Log Diff'}, 'yaxis': {'title': 'Count'}, 'bargroupgap': 0.1,}}
    iplot(fig)
    
def plot_empirical_cdf(ts, plot_title, **kwargs): 
    logdiffs = get_diffs(ts)
    fig = {'data':
            [{'x': logdiffs.values,
              'type': 'histogram', 'cumulative': {'enabled': True},  
            'histnorm': 'probability', 'name': "distribution of logarithmic differences", },],
          'layout': 
          {'title': plot_title, 
            'xaxis': {'title': 'Log Diff'}, 'yaxis': {'title': 'Empirical Probability'}, 'bargroupgap': 0.1,}}
    iplot(fig)

      
def plot_scatter(ts, lag, **kwargs): 
    logdiffs = get_diffs(ts)
    if kwargs["diff_type"] == "abs_diffs":
        values = get_diffs(ts,"abs_diffs")
        xaxis_title = "Absolute Diffs "
    elif  kwargs["diff_type"] == "relative_diffs": 
        values = get_diffs(ts,"relative_diffs")
        xaxis_title = "Relative Diffs"
    else: 
        values = ts.values
        xaxis_title = "Price"
    fig = {'data':
              [{'x': values, 'y': logdiffs.rolling(window=lag,center=False).var().values,
               'mode': 'markers', 'marker': {'opacity': 0.25},}],
           'layout': {
               'title': kwargs["plot_title"], 
               'xaxis': {'title': xaxis_title}, 'yaxis': {'title': kwargs["volatility_label"]},}} 
    iplot(fig)
   
        
def get_X(ts, lag): 
    logdiffs = get_diffs(ts)
    monthly_vol = logdiffs.rolling(window=lag,center=False).var()
    X = list(zip(ts.values, monthly_vol.values))
    res = X[len(monthly_vol.index[np.isnan(monthly_vol)]):]
    return res

def cluster_vol_vs_price(ts, **kwargs):
 # kwargs: method, plot_title, number_of_clusters, volatility_label
   
    clusters = KMeans(n_clusters=kwargs["number_of_clusters"])
    X = get_X(ts, kwargs["vol_window"])
    clusters.fit(X)
    print('Silhouette cofficient for {} clusters: {}'.format(kwargs["number_of_clusters"],
                    metrics.silhouette_score(X, clusters.labels_, metric='euclidean')))
  
    if len(colors) < kwargs["number_of_clusters"]: 
        kwargs["number_of_clusters"] = len(colors)
        
    fig = {
        'data':[ {
            'x': [X[i][0] for i in range(len(clusters.labels_)) if clusters.labels_[i]==k],
            'y': [X[i][1] for i in range(len(clusters.labels_)) if clusters.labels_[i]==k],
            'mode': 'markers', 'marker': dict(color=colors[k], opacity=0.25),
            'name': "Cluster " + str(k+1), } for k in range(kwargs["number_of_clusters"])],
        'layout': {
            'title': kwargs["plot_title"], 
            'xaxis': {'title': 'Price'}, 'yaxis': {'title': kwargs["volatility_label"]} ,}
    }
        
    iplot(fig) 
 
    return clusters   
