import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv
from scipy import stats
# Keicimen = 0
# Besni = 1

# ----BEFORE CLEANING----
names = ['area', 'major', 'minor', 'ecc', 'convex', 'extent', 'perimeter', 'class']
dataset = read_csv('Raisin_Dataset_CSV.csv', header=0, names=names)

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


# Clean the Data
threshold_z = 3
outlier_indices = []
for col in dataset:
    if col != "class":
        z = np.abs(stats.zscore(dataset[col]))
        outlier_indices_col = np.where(z > threshold_z)[0]
        outlier_indices.extend(outlier_indices_col)
dataset = dataset.drop(outlier_indices)



# ----AFTER CLEANING----

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
