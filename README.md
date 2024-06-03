# CodeAlpha_DATA-EXPLORATION-IRIS-FLOWER-CLASSIFICATION
## Task 2  :  DATA EXPLORATION : IRIS FLOWER CLASSIFICATION


### Load the Iris dataset and explore its structure. Check for missing values and handle them if necessary. Visualize the data using plots and graphs to understand the distribution of each feature

#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay

#### Load the dataset

data = pd.read_csv("Iris.csv")

data

data.describe()

data.info()

#### Data cleaning and pre processing

data.isna().sum()

data["Species"] = data["Species"].str.replace("Iris-","")

data

#### EDA

sns.pairplot(data, hue = 'Species')

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
sns.histplot(data=data, x='SepalLengthCm', hue='Species', kde=True, ax=axs[0, 0])
sns.histplot(data=data, x='SepalWidthCm', hue='Species', kde=True, ax=axs[0, 1])
sns.histplot(data=data, x='PetalLengthCm', hue='Species', kde=True, ax=axs[1, 0])
sns.histplot(data=data, x='PetalWidthCm', hue='Species', kde=True, ax=axs[1, 1])
plt.suptitle('Distribution of Features by Species')
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
sns.boxplot(data=data, x='Species', y='SepalLengthCm', ax=axs[0, 0])
sns.boxplot(data=data, x='Species', y='SepalWidthCm', ax=axs[0, 1])
sns.boxplot(data=data, x='Species', y='PetalLengthCm', ax=axs[1, 0])
sns.boxplot(data=data, x='Species', y='PetalWidthCm', ax=axs[1, 1])
plt.suptitle('Boxplot of Features by Species')
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
sns.violinplot(data=data, x='Species', y='SepalLengthCm', ax=axs[0, 0])
sns.violinplot(data=data, x='Species', y='SepalWidthCm', ax=axs[0, 1])
sns.violinplot(data=data, x='Species', y='PetalLengthCm', ax=axs[1, 0])
sns.violinplot(data=data, x='Species', y='PetalWidthCm', ax=axs[1, 1])
plt.suptitle('Violin Plots of Features by Species')
plt.show()

sns.lmplot(data=data, x='PetalLengthCm', y='PetalWidthCm', hue='Species', height=7)
plt.title('Scatter Plot with Regression Line (Petal Length vs Petal Width)')
plt.show()

x= data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

x

y = data['Species']

y

y.value_counts()

#### Splitting the dataset into testing and training

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x_train

x_test

### Train with Logistic Regression model

model = LogisticRegression()

model.fit(x_train,y_train)

predict = model.predict(x_test)

predict

cm = confusion_matrix(predict,y_test)
cm

cm = ConfusionMatrixDisplay(cm)
cm.plot()
plt.title("Confusion Matrix by Logistic Regression")

report = classification_report(predict,y_test)
print(report)

#### Model Evaluation

new_data = x.iloc[[20]]
pred = model.predict(new_data)

pred

result = y.iloc[73]
result

#### Run the Model

sl = float(input("Enter the Sepal length :"))
sw = float(input("Enter the Sepal width :"))
pl = float(input("Enter the Petal length :"))
pw = float(input("Enter the Petal width :"))
new_data = [[sl,sw,pl,pw]]
pred = model.predict(new_data)
print("The species of flower with repect of follwing measurements is :",pred[0])

## The End

