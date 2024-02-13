###
# This code file contains an import of all of the libraries we expect to use in the final project at the top. 
# Unused imports will be removed in the phase 4 submission. 
# Then the dataset is loaded in as a CSV file downloaded from the same directory.
# We summarize the dataset and replace the Class names with 0 and 1 before starting the cleaning process we check for missing values.
# This particular dataset had no missing values so the only step of the cleaning process that changes the dataset is removal of outliers.
# We include data visualizations before and after removing the outliers based off a z score of 3.
###


# Load libraries
# remove unused imports in final project if code is turned in
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import sklearn.metrics as met


# Keicimen = 0
# Besni = 1

# ----BEFORE CLEANING----
names = ['area', 'major', 'minor', 'ecc', 'convex', 'extent', 'perimeter', 'class']
dataset = read_csv('Raisin_Dataset_CSV.csv', header=0, names=names)
# summarize the dataset
print(dataset.describe())

# replace class names with 0 or 1 values for classification
dataset = dataset.replace('Kecimen', 0)
dataset = dataset.replace('Besni', 1)
print(dataset.sample(20))

# count the missing values
print(dataset.isnull().sum())

# locate the column with the 99999 values
num_missing = (dataset == 99999).sum()
print(num_missing)
# find and print the indices
for col in dataset:
    if num_missing[col] != 0:
        indices = dataset[dataset[col] == 99999].index
        print(col + ":" + str(list(indices)))

# locate the column with the 0 values
num_missing = (dataset == 0).sum()
print(num_missing)
# find and print the indices
# we can ignore the 0 values in class because 0 is a valid value
for col in dataset:
    if num_missing[col] != 0:
        indices = dataset[dataset[col] == 0].index
        print(col + ":" + str(list(indices)))


# ----DATA PLOTS----

# Box Plot
dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()

# Density Curve
dataset.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
plt.title("Data Density Before Cleaning")
plt.show()


# Histograms
# Area Histogram
x = dataset['area']
plt.hist(x)
plt.title("Area Histogram")
plt.show()
# Major Axis Length Histogram
x = dataset['major']
plt.hist(x)
plt.title("Major Axis Length Histogram")
plt.show()
# Minor Axis Length Histogram
x = dataset['minor']
plt.hist(x)
plt.title("Minor Axis Length Histogram")
plt.show()
# Eccentricity Histogram
x = dataset['ecc']
plt.hist(x)
plt.title("Eccentricity Histogram")
plt.show()
# ConvexArea Histogram
x = dataset['convex']
plt.hist(x)
plt.title("Convex Area Histogram")
plt.show()
# Extent Histogram
x = dataset['extent']
plt.hist(x)
plt.title("Extent Histogram")
plt.show()
# Perimeter Histogram
x = dataset['perimeter']
plt.hist(x)
plt.title("Perimeter Histogram")
plt.show()


# Correlation Matrix
correlations = dataset.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,8,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# ----Clean the Data----
threshold_z = 3
outlier_indices = []
for col in dataset:
    if col != "class":
        z = np.abs(stats.zscore(dataset[col]))
        outlier_indices_col = np.where(z > threshold_z)[0]
        outlier_indices.extend(outlier_indices_col)
print(outlier_indices)
dataset = dataset.drop(outlier_indices)
print(dataset.describe())


# ----DATA PLOTS AFTER CLEANING----

# Box Plot
dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()



# Density Plot
dataset.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
plt.title("Data Density After Cleaning")
plt.show()


# Histograms
# Area Histogram
x = dataset['area']
plt.hist(x)
plt.title("Area Histogram After Cleaning")
plt.show()
# Major Axis Length Histogram
x = dataset['major']
plt.hist(x)
plt.title("Major Axis Length Histogram After Cleaning")
plt.show()
# Minor Axis Length Histogram
x = dataset['minor']
plt.hist(x)
plt.title("Minor Axis Length Histogram After Cleaning")
plt.show()
# Eccentricity Histogram
x = dataset['ecc']
plt.hist(x)
plt.title("Eccentricity Histogram After Cleaning")
plt.show()
# ConvexArea Histogram
x = dataset['convex']
plt.hist(x)
plt.title("Convex Area Histogram After Cleaning")
plt.show()
# Extent Histogram
x = dataset['extent']
plt.hist(x)
plt.title("Extent Histogram After Cleaning")
plt.show()
# Perimeter Histogram
x = dataset['perimeter']
plt.hist(x)
plt.title("Perimeter Histogram After Cleaning")
plt.show()


# Correlation Matrix
correlations = dataset.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,8,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
