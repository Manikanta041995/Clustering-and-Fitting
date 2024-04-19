# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:13:57 2024

@author: Sasidhar
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
from matplotlib.colors import ListedColormap

def df_info(df):
    """This function takes data frame as input and returns
       structure of the data frame such as columns,head,tail
       ,transpose,summary
       Parameters:
           df(DataFrame): Data containg 'CO2 from solid fuel' and 'CO2 from solid fuel' data
           """
    print('Columns of the Data Frame\n')
    print(df.columns)
    print('\n\n')
    print('The top values of Data Frame\n')
    print(df.head())
    print('\n\n')
    print('The bottom values of Data Frame\n')
    print(df.tail())
    print('\n\n')
    print(f'The size of the data frame : {df.size}\n')
    print(f'The shape of the data frame : {df.shape}\n')
    print('The transpose of Data Frame\n')
    print(df.T)
    print('\n\n')
    print('summary of the Data Frame\n')
    print(df.info(verbose = True))

def analysis(df):
    """Performs a brief analysis of numerical columns and 
    returns kurtosis and skewness.
    Parameters:
        df(DataFrame): Data containg 'CO2 from solid fuel' and 'CO2 from solid fuel' data
        
    returns:
        list : list of kurtosis and skewness"""
    print('Brief analysis of Numerical Columns')
    print(df.describe())
    #calucating the skewness
    skew = df.skew(numeric_only = True)
    print('\n')
    print('The skewness of Numerical Columns')
    print(skew)
    #calucating the kurtosis
    kurt = df.kurtosis(numeric_only = True)
    print('\n')
    print('The kurtosis of Numerical Columns')
    print(kurt)
    return [kurt, skew]

def scatter_plot(df, image_name):
    """Creates a scatter plot for 'CO2 from liquid fuel' vs 'CO2 from solid fuel'.
    Parameters:
        df(DataFrame): Data containg 'CO2 from solid fuel' and 'CO2 from solid fuel' data
        image_name: Title and image name of the plot
    """
    plt.figure()
    # Scatter plot with 'Price' on the x-axis and 'Point' on the y-axis
    plt.scatter(df['CO2 from liquid fuel'],df['CO2 from solid fuel'])
    plt.xlabel('CO2 from liquid fuel')
    plt.ylabel('CO2 from solid fuel')
    plt.title(image_name)
    # Set x-axis scale to logarithmic for better visualization
    plt.xscale('log')
    plt.savefig(image_name,dpi='figure',bbox_inches='tight')
    plt.show()

def box_plot(df, image_name):
    """Creates a box plot for outliers in numeric columns.
    Parameters:
        df(DataFrame): Data containg 'CO2 from solid fuel' and 'CO2 from solid fuel' data
        image_name: Title and image name of the plot"""
    
    columns = ['CO2 from liquid fuel', 'CO2 from solid fuel', 'CO2 emission']
    plt.figure()
    # Generate a box plot for the specified columns in the DataFrame
    plt.boxplot(df[columns])
    # Set y-axis scale to logarithmic for better visualization
    plt.yscale('log')
    # Set x-axis ticks and labels for each column
    plt.xticks([1,2,3],columns)
    plt.ylabel('Frequency')
    plt.title('Box plot for outliers in Numeric cols')
    plt.savefig(image_name,dpi='figure',bbox_inches='tight')
    plt.show()
    
def hist_plot(df, image_name):
    """Plots histograms for 'Price', 'Rating', and 'No. of People rated'.
    Parameters:
        df(DataFrame): Data containg 'CO2 from solid fuel' and 'CO2 from solid fuel' data
        image_name: Title and image name of the plot"""
    #setting the backgroud to whitgrid.
    sns.set_style("whitegrid")
    plt.figure(figsize=(18, 14), dpi=200)
    #creating 3 subplots in frist column.
    plt.subplot(3, 1, 1)
    #ploting histogram in frist row for CO2 from liquid fuel
    sns.histplot(df['CO2 from liquid fuel'], bins=10, kde=True)
    plt.ylabel('Frequency')
    plt.title('Distribution of CO2 from liquid fuel')
    plt.subplot(3, 1, 2)
    #ploting histogram in second row for CO2 from solid fuel
    sns.histplot(df['CO2 from solid fuel'], bins=30, kde=True)
    plt.ylabel('Frequency')
    plt.title('Distribution of CO2 from solid fuel')
    plt.savefig(image_name,dpi='figure',bbox_inches='tight')
    plt.show()
    plt.subplot(3, 1, 3)
    #ploting histogram in second row for CO2 emission
    sns.histplot(df['CO2 emission'], bins=30, kde=True)
    plt.ylabel('Frequency')
    plt.title('Distribution of CO2 from solid fuel')
    plt.savefig(image_name,dpi='figure',bbox_inches='tight')
    plt.show()
    
def heatmap(df, image_name):
    """Plots a heatmap for the correlation between numeric columns.
    Parameters:
        df(DataFrame): Data containg 'CO2 from solid fuel' and 'CO2 from solid fuel' data
        image_name: Title and image name of the plot"""
    plt.figure()
    # Creating Heatmap for Numeric columns
    sns.heatmap(df.corr(),annot=True,vmin=-1, vmax=1, 
                 annot_kws={'fontsize':8, 'fontweight':'bold'})
    plt.title('Correlation heatmap between numeric columns')
    plt.savefig(image_name,dpi='figure',bbox_inches='tight')
    plt.show()


def silhouette_score(df):
    """ Calculates the Silhoutee score with the range of 2 - 10
    Parameters:
        df(DataFrame): Data containg 'CO2 from solid fuel' and 'CO2 from solid fuel' data
        
    returns :
        silhouette_scores(list): list of silhoute scores"""
    # Loop over the number of clusters and calculate silhouette scores
    silhouette_scores = []
    for ncluster in range(2, 10):
        kmeans = KMeans(n_clusters=ncluster)
        kmeans.fit(df)
        labels = kmeans.labels_
        score = skmet.silhouette_score(df, labels)
        silhouette_scores.append(score)
        print(f"Number of Clusters: {ncluster}, Silhouette Score: {score}")
    return silhouette_scores

def line_plot(x,y,x_label,y_label, image_name):
    """ plots the line plot"""
    # Plot the line plot
    plt.figure()
    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(image_name)
    plt.savefig(image_name)
    plt.show()
    
def plot_fitted(labels, xy, xkmeans, ykmeans, centre_labels, image_name):
    """
    Plots clustered data as a scatter plot with determined centres shown
    parameters: 
        labels(array like): labels from Kmeans
        xy(array like): inverted scaler
        xkmeans(array like): x value of kmeans
        ykmeans(array like): y values of kmeans
        centre_labels(array like): predicted values of clusters 
    """
    colours = plt.cm.Set1(np.linspace(0, 1, len(np.unique(labels))))
    cmap = ListedColormap(colours)
    
    fig, ax = plt.subplots(dpi=144)
    s = ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap=cmap, marker='o', label='Data')

    ax.scatter(xkmeans, ykmeans, c=centre_labels, cmap=cmap, marker='x', s=100, label='Estimated Centres')

    cbar = fig.colorbar(s, ax=ax)
    cbar.set_ticks(np.unique(labels))
    ax.legend()
    ax.set_xlabel("CO2 from liquid fuel")
    ax.set_ylabel("CO2 from solid fuel")
    ax.set_xscale('log')
    plt.savefig(image_name+"predicted")
    plt.show()

def k_means(df,n):
   """
    Perform K-means clustering on the given dataset.

    Parameters:
    - df (DataFrame): The input DataFrame containing the data to be clustered.
    - n (int): The number of clusters to create.

    Returns:
    - kmeans (KMeans): The trained KMeans clustering model.
    """
   #Set the number of clusters to 2
   kmeans = KMeans(n_clusters=n)
   kmeans.fit(df)
   return kmeans

def logistic(t, n0, g, t0):
    """
    Calculates the logistic function with scale factor n0 and growth rate g.

    Parameters:
        t (array-like): Independent variable values.
        n0 (float): Scale factor.
        g (float): Growth rate.
        t0 (float): Inflection point.

    Returns:
        array-like: Logistic function values.
    """
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f


def plot_forecast(year, data, param, covar, predictions=None):
    """
    Plots the forecasted CO2 emissions along with the confidence \
        interval and predictions.

    Parameters:
        year (array-like): Years for the forecast.
        data (DataFrame): Data containing 'Year' and 'CO2 emissions' columns.
        param (array-like): Parameters for the logistic function.
        covar (array-like): Covariance matrix from curve_fit.
        predictions (array-like, optional): Predicted CO2 emissions values. \
            Default is None.

    Returns:
        None
    """
    forecast = logistic(year, *param)

    # Calculate confidence interval
    stderr = np.sqrt(np.diag(covar))
    conf_interval = 1.96 * stderr
    upper = logistic(year, *(param + conf_interval))
    lower = logistic(year, *(param - conf_interval))

    # Plot the result
    plt.figure()
    plt.plot(data["Year"], data["CO2 emissions"], label="CO2 emissions")
    plt.plot(year, forecast, label="Forecast")
    plt.fill_between(year, upper, lower, color='purple', alpha=0.2, \
                     label="95% Confidence Interval")
    if predictions is not None:
        plt.plot(year_pred, predictions, 'ro-', label='Predictions')
    plt.xlabel("Year")
    plt.ylabel("CO2 emissions")
    plt.title("Logistic forecast for China")
    plt.legend()
    plt.savefig("Logistic forecast for China")
    plt.show()
def clustering(df_cluster, x_label, y_label, image_name):
    """
   Perform K-means clustering on the given dataset and visualize the clusters.

   Parameters:
   - df_cluster (DataFrame): The input DataFrame containing the data to be clustered.
   - x_label (str): The label for the x-axis in the plot.
   - y_label (str): The label for the y-axis in the plot."""
    scaler = StandardScaler()
    df_cluster = scaler.fit_transform(df_cluster)

    silhouette_scores = silhouette_score(df_cluster)    
    # Plot the silhouette scores
    line_plot(range(2, 10), silhouette_scores, "Number of Clusters", "Silhouette Score", 
              f"Silhouette Score for Different Numbers of Clusters_{image_name[len(image_name)-6:len(image_name)]}")


    # Find the optimal number of clusters with the highest silhouette score
    optimal_ncluster = np.argmax(silhouette_scores) + 2
    print(f"\nOptimal Number of Clusters: {optimal_ncluster}")  
    inv_scaler = scaler.inverse_transform(df_cluster)
    kmeans = k_means(df_cluster, optimal_ncluster)
    labels = kmeans.labels_
    cen = scaler.inverse_transform(kmeans.cluster_centers_)
    xcen = cen[:, 0]
    ycen = cen[:, 1]
    cen_labels = kmeans.predict(kmeans.cluster_centers_)

    plot_fitted(labels, inv_scaler, xcen, ycen, cen_labels, image_name)

    cen = kmeans.cluster_centers_
    xcen = cen[:, 0]
    ycen = cen[:, 1]

    #Plot the clusters and cluster centers
    plt.figure(figsize=(6, 5))
    cm = plt.cm.get_cmap('tab10')
    for i, label in enumerate(np.unique(labels)):

        plt.scatter(df_cluster[labels == label, 0], \
                    df_cluster[labels == label, 1], 10, \
                        label=f"Cluster {label}", cmap=cm, alpha=0.7)
    plt.scatter(xcen, ycen, 45, "k", marker="d", label="Cluster centers")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Kmeans clustering")
    plt.legend()#
    plt.savefig(image_name)
    plt.show()

# Load and preprocess the data for clustering
liquid = pd.read_csv("CO2 emissions from liquid fuel consumption (kt).csv", \
                     skiprows=4)
solid = pd.read_csv("CO2 emissions from solid fuel consumption (kt).csv", \
                    skiprows=4)
CO2 = pd.read_csv("CO2 emissions(kt).csv", skiprows=4)  

# Drop rows with NaN values in 2015
liquid = liquid[liquid["2015"].notna()]
solid = solid.dropna(subset=["2015"])


# Merge the datasets for 2015
liquid2015 = liquid[["Country Name", "Country Code", "2015"]].copy()
solid2015 = solid[["Country Name", "Country Code", "2015"]].copy()
co2015 = CO2[["Country Name", "Country Code", "2015"]].copy()
df_2015 = pd.merge(liquid2015, solid2015, on="Country Name", how="outer")
df_2015 = pd.merge(df_2015, co2015, on="Country Name", how="outer")
df_2015 = df_2015.dropna()  # Drop entries with missing values4
df_2015 = df_2015.rename(columns={"2015_x": "CO2 from liquid fuel", \
                                  "2015_y": "CO2 from solid fuel",
                                  "2015": "CO2 emission"})
df_info(df_2015)
analysis(df_2015)
scatter_plot(df_2015, 'CO2 from liquid fuel vs CO2 from solid fuel')
box_plot(df_2015, 'Box plot for Numeric columns')
hist_plot(df_2015, 'Histplot for distribution')
df_copy = df_2015.copy()
df_copy = df_copy.drop(columns = ['Country Name', 'Country Code_x', 
                                  'Country Code_y', 'Country Code'])
heatmap(df_copy, 'Confusion matrix for co2 from liquid and solid fuel')

# Perform clustering
df_cluster_solid = df_2015[["CO2 emission", "CO2 from solid fuel"]].copy()
df_cluster_liquid = df_2015[["CO2 emission", "CO2 from liquid fuel"]].copy()

clustering(df_cluster_liquid, "CO2 emissions", "CO2 from liquid fuel",
           "Kmeans clustering_liquid")
clustering(df_cluster_solid, "CO2 emissions", "CO2 from solid fuel", 
           "Kmeans clustering_solid")
CO2 = CO2.set_index('Country Name', drop=True)
CO2 = CO2.loc[:, '1960':'2021']
CO2 = CO2.transpose()
CO2 = CO2.loc[:, 'China']
df = CO2.dropna(axis=0)

df_co2 = pd.DataFrame()
df_co2['Year'] = pd.DataFrame(df.index)
df_co2['CO2 emissions'] = pd.DataFrame(df.values)

#Convert year column to numeric
df_co2["Year"] = pd.to_numeric(df_co2["Year"])

#Convert year column to numeric
df_co2["Year"] = pd.to_numeric(df_co2["Year"])

#Fit the logistic function to the data
param, covar = curve_fit(logistic, df_co2["Year"], \
                          df_co2["CO2 emissions"], p0=(1.2e12, 0.03, 1990.0), \
                              maxfev=10000)

#Generate years for the forecast
year = np.arange(1960, 2031)

#Plot the forecast
plot_forecast(year, df_co2, param, covar)

# Generate predictions for the next 10 years
year_pred = np.arange(2022, 2031)
predictions = logistic(year_pred, *param)

# Plot the forecast and show predictions
plot_forecast(year, df_co2, param, covar, predictions)
