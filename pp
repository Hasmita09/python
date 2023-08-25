import pandas as pd

# Read the first CSV file
data1 = pd.read_csv(r"C:\Users\hasmi\OneDrive\Desktop\google play store.zip")

# Read the second CSV file
data2 = pd.read_csv(r"C:\Users\hasmi\OneDrive\Desktop\playp.zip")

# Print column names for data1 and data2 to identify common column
print("Columns in data1:", data1.columns)
print("Columns in data2:", data2.columns)

# Merge the two dataframes based on the identified common column
data = pd.merge(data1, data2)

# Perform further analysis or operations on the merged_data
print(data.head())

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
import random
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv(r"C:\Users\hasmi\OneDrive\Documents\google play store.zip")
data.info()
data.shape
data.isnull().any()
data.isnull().sum()
data = data.dropna()
data.isnull().any()
data.shape
data["Size"] = [float(i.split('M')[0]) if isinstance(i, str) and 'M' in i else float(0) for i in data["Size"]]
data.head()
data["Size"] = 1000 * data["Size"]
data
data.info()
data["Reviews"] = data["Reviews"].astype(float)
data.info()
data["Installs"] = [ float(i.replace('+','').replace(',', '')) if '+' in i or ',' in i else float(0) for i in data["Installs"] ]
data.head()
data.info()
data["Installs"] = data["Installs"].astype(int)
data.info()
data['Price'] = [ float(i.split('$')[1]) if '$' in i else float(0) for i in data['Price'] ]
data.head()
data.info()
data["Price"] = data["Price"].astype(int)
data.info()
data.shape
data.drop(data[(data['Reviews'] < 1) & (data['Reviews'] > 5 )].index, inplace = True)
data.shape
data.drop(data[data['Installs'] < data['Reviews'] ].index, inplace = True)
data.shape
data.drop(data[(data['Type'] =='Free') & (data['Price'] > 0 )].index, inplace = True)
data.shape
sns.set(rc={'figure.figsize':(12,8)})
sns.boxplot(data['Price'])
sns.boxplot(data['Reviews'])
sns.boxplot(data['Ratings'])
sns.boxplot(data['Size'])
more = data.apply(lambda x : True
            if x['Price'] > 200 else False, axis = 1) 
more_count = len(more[more == True].index)
data.shape
data.drop(data[data['Price'] > 200].index, inplace = True)
data.shape
data.drop(data[data['Reviews'] > 2000000].index, inplace = True)
data.shape
numeric_columns = ["Size", "Installs", "Price"]
quantiles = [0.1, 0.25, 0.5, 0.7, 0.9, 0.95, 0.99]

quantile_data = data[numeric_columns].quantile(quantiles)

print(quantile_data)
# dropping more than 10000000 Installs value
data.drop(data[data['Installs'] > 10000000].index, inplace = True)
data.shape
sns.scatterplot(x='Rating',y='Price',data=data)
#Paid apps are higher ratings comapred to free apps.
sns.scatterplot(x='Rating',y='Size',data=data)
#Heavier apps are rated better.
sns.scatterplot(x='Rating',y='Reviews',data=data)
#More reviews makes app rating better.
sns.barplot(x="Rating", y="Content Rating", data=data)
#Apps categorized as "Everyone" tend to exhibit more unfavorable ratings when compared to other sections due to the presence of numerous outlier values. On the contrary, applications classified as "18+" generally showcase higher ratings.
sns.barplot(x="Rating", y="Category", data=data)
#Events category has best ratings compare to others.
inp1 = data
inp1.head()
reviewskew = np.log1p(inp1['Reviews'])
inp1['Reviews'] = reviewskew
reviewskew.skew()
installsskew = np.log1p(inp1['Installs'])
inp1['Installs']
installsskew.skew()
inp1.head()
inp1.shape
inp2 = inp1
#Applying Dummy EnCoding on Column "Category"
#get unique values in Column "Category"
inp2.Category.unique()
inp2.Category = pd.Categorical(inp2.Category)

x = inp2[['Category']]
del inp2['Category']

dummies = pd.get_dummies(x, prefix = 'Category')
inp2 = pd.concat([inp2,dummies], axis=1)
inp2.head()
inp2.shape
#Applying Dummy EnCoding on Column "Genres"
#get unique values in Column "Genres"
inp2["Genres"].unique()
# Since, There are too many categories under Genres. Hence, we will try to reduce some categories which have very few samples under them and put them under one new common category i.e. "Other".

lists = []
for i in inp2.Genres.value_counts().index:
    if inp2.Genres.value_counts()[i]<20:
        lists.append(i)
inp2.Genres = ['Other' if i in lists else i for i in inp2.Genres]
inp2["Genres"].unique()
inp2.Genres = pd.Categorical(inp2['Genres'])
x = inp2[["Genres"]]
del inp2['Genres']
dummies = pd.get_dummies(x, prefix = 'Genres')
inp2 = pd.concat([inp2,dummies], axis=1)
inp2.head()
inp2.shape
#Let's apply Dummy EnCoding on Column "Content Rating"
#get unique values in Column "Content Rating"
inp2["Content Rating"].unique()
inp2['Content Rating'] = pd.Categorical(inp2['Content Rating'])
data.shape
x = inp2[['Content Rating']]
del inp2['Content Rating']
dummies = pd.get_dummies(x, prefix = 'Content Rating')
inp2 = pd.concat([inp2,dummies], axis=1)
inp2.head()

from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as mse
d1 = inp2
X = d1.drop('Rating',axis=1)
y = d1['Rating']

Xtrain, Xtest, ytrain, ytest = tts(X,y, test_size=0.3, random_state=5)
reg_all = LR()
reg_all.fit(Xtrain,ytrain)
R2_train = round(reg_all.score(Xtrain,ytrain),3)
print("The R2 value of the Training Set is : {}".format(R2_train))
R2_test = round(reg_all.score(Xtest,ytest),3)
print("The R2 value of the Testing Set is : {}".format(R2_test))
