#Importing Libraries
#First of all we will import known mandatory libraries
import numpy as np
import pandas as pd
import pylab as pl
import sklearn
from sklearn.model_selection import train_test_split
#Then import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#Loading the dataset.
#We use pandas to read the dataset and preprocess it
telecom_dataset = pd.read_csv("Telecom Churn Dataset.csv")
#Printing the whole dataset
print("The whole dataset")
print(telecom_dataset)

#Exploratory Data Analysis
#Understanding the data
print("The first five rows ")
print(telecom_dataset.head())
print("The last five rows")
print(telecom_dataset.tail())
print("The number of rows and columns")
print(telecom_dataset.shape)
print("Column names")
print(telecom_dataset.columns)
print("The datatypes of each column")
print(telecom_dataset.dtypes)

#Basic information
print(telecom_dataset.info())

#Describing the data
print(telecom_dataset.describe())
#Find the duplicates
print("The number unique values of each column in the dataset")
print(telecom_dataset.nunique())
#print("The number of duplicate values")
print(telecom_dataset.duplicated().sum())
print("The actual unique values of each column in the dataset")
print(telecom_dataset["ID"].unique())
print(telecom_dataset["Gender"].unique())
print(telecom_dataset["SeniorCitizen"].unique())
print(telecom_dataset["Married"].unique())
print(telecom_dataset["Tenure"].unique())
print(telecom_dataset["PhoneService"].unique())
print(telecom_dataset["MultipleLines"].unique())
print(telecom_dataset["InternetService"].unique())
print(telecom_dataset["TechSupport"].unique())
print(telecom_dataset["StreamingTV"].unique())
print(telecom_dataset["StreamingMovies"].unique())
print(telecom_dataset["Contract"].unique())
print(telecom_dataset["PaperlessBilling"].unique())
print(telecom_dataset["PaymentMethod"].unique())
print(telecom_dataset["MonthlyCharges"].unique())
print(telecom_dataset["TotalCharges"].unique())
print(telecom_dataset["Churn"].unique())

#Cleaning the data
print("Finding the null and non-null values of the columns in the dataset")
print(telecom_dataset.isnull())
print(telecom_dataset.isnull().sum())
print(telecom_dataset.notnull().sum())

#Visualising the dataset

#Pie Charts
#Initializing the data
labels = ["No", "Yes"]
print(telecom_dataset.groupby(["Churn"])["Churn"].count())
values = (list(telecom_dataset.groupby(["Churn"])["Churn"].count()))

explode = [0.1, 0.5]

#Plotting the data
plt.pie(values, labels=labels, explode=explode, autopct='%1.2f%%', shadow=True)
plt.title("PERCENTAGE OF CUSTOMERS WHO HAVE CHURNED AND NOT CHURNED",fontweight="bold",fontsize=20)
plt.legend(title="YES OR NO")
plt.show()
#Initializing the data
labels = ["Female", "Male"]
mycolors = ["cyan","darkcyan"]
print(telecom_dataset.groupby(["Gender"])["Gender"].count())
values = list(telecom_dataset.groupby(["Gender"])["Gender"].count())

explode = [0.1, 0.5]

#Plotting the data
plt.pie(values, labels=labels, explode=explode, autopct='%1.2f%%', shadow=True,colors=mycolors)
plt.title("PERCENTAGE OF MALE AND FEMALE CUSTOMERS",fontweight ="bold",fontsize=20)
plt.legend(title="MALE OR FEMALE")
plt.show()

#Initializing the data
labels = [0, 1]
mycolors2 = ["limegreen","darkgreen"]
print(telecom_dataset.groupby(["SeniorCitizen"])["SeniorCitizen"].count())
values = list(telecom_dataset.groupby(["SeniorCitizen"])["SeniorCitizen"].count())

explode = [0.1, 0.5]

#Plotting the data
plt.pie(values, labels=labels, explode=explode, autopct='%1.2f%%', shadow=True, colors=mycolors2)
plt.title("PERCENTAGE OF SENIOR CITIZENS AND NON-SENIOR CITIZENS",fontweight ="bold",fontsize=20)
plt.legend(title="SENIOR CITIZEN OR NON-SENIOR CITIZEN")
plt.show()

#Countplot
plt.style.use("Solarize_Light2")
countplot = sns.countplot(x="Churn", data=telecom_dataset, palette="inferno")
countplot.set_ylabel("Count")
countplot.set_title("CUSTOMER CHURN AND RETENTION COUNT", fontdict={'size': 20, 'weight': 'bold'})
plt.show()


plt.style.use("ggplot")
countplot2 = sns.countplot(x="Churn", hue="Gender", data=telecom_dataset, palette="spring")
countplot2.set_ylabel("Count")
countplot2.set_title("CHURN COUNT ON THE BASIS OF GENDER", fontdict={'size': 20, 'weight': 'bold'})
plt.show()

plt.style.use("dark_background")
df =sns.countplot(x="Churn", hue="Contract", data=telecom_dataset, palette="mako")
df.set_ylabel("Number of customers")
df.set_title("COUNT OF CUSTOMER'S CONTRACT DURATION ON THE BASIS OF CUSTOMER'S CHURN",fontdict={'size': 20, 'weight': 'bold'})
plt.show()
del telecom_dataset["ID"]
#Pairplot
pairplot = sns.pairplot(data=telecom_dataset, hue="Churn",palette="Spectral")

pairplot.fig.suptitle("TELECOMMUNICATION DATASET'S PAIRPLOT ON THE BASIS OF CUSTOMER'S CHURN",fontweight="bold",fontsize=20)
plt.show()

pairplot2 = sns.pairplot(data=telecom_dataset)
pairplot2.fig.suptitle("TELECOMMUNICATION DATASET'S PAIRPLOT",fontweight="bold",fontsize=20)
plt.show()

#Histogram
telecom_dataset.hist(bins=30,color="palevioletred")
pl.suptitle("HISTOGRAM COLLECTION OF NUMERICAL FEATURES",fontweight="bold",fontsize=20)
plt.show()

#Statistics
print(telecom_dataset.groupby("Churn").MonthlyCharges.describe().round(0))
print(telecom_dataset['Churn'].describe())

#Cleaning Data
#Removing Gender, CustomerID,tenure as they are not useful
col = ["Gender", "Tenure"]
telco_data = telecom_dataset.drop(col, axis=1)
#Countplot
for i, predictor in enumerate(telco_data.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    ax = sns.countplot(data =telco_data, x=predictor, hue='Churn')
    ax.set_ylabel("Count")
    if predictor == "PaymentMethod":
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
        plt.tight_layout()
        plt.show()
    else:
        plt.tight_layout()
        plt.show()

categorical_features = ["Gender", "SeniorCitizen", "Married", "PhoneService", "MultipleLines", "InternetService", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
numerical_features = ["Tenure", "MonthlyCharges", "TotalCharges"]
target = "Churn"
print(telecom_dataset.skew(numeric_only=True))

print(telecom_dataset.corr(numeric_only=True))
print(telecom_dataset[numerical_features].describe())
print(telecom_dataset[numerical_features].hist(bins=30, figsize=(10, 7)))
pl.suptitle("HISTOGRAM COLLECTION OF NUMERICAL FEATURES",fontweight ="bold",fontsize=20)
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
telecom_dataset[telecom_dataset.Churn == "No"][numerical_features].hist(bins=30, color="dodgerblue", alpha=0.5, ax=ax)
telecom_dataset[telecom_dataset.Churn == "Yes"][numerical_features].hist(bins=30, color="lawngreen", alpha=0.5, ax=ax)
pl.suptitle("HISTOGRAM COLLECTION OF NUMERICAL FEATURES",fontweight ="bold",fontsize=20)
plt.show()
#Target variable distribution
td = telecom_dataset[target].value_counts().plot(kind='bar',color="mediumorchid").set_title('CHURNED',fontweight ="bold",fontsize=20)
plt.xlabel("Churn")
plt.ylabel("Count")

plt.show()
#Outliers Analysis with IQR Method
x = ['Tenure', 'MonthlyCharges']
def count_outliers(data, col):
        q1 = data[col].quantile(0.25,interpolation='nearest')
        q2 = data[col].quantile(0.5,interpolation='nearest')
        q3 = data[col].quantile(0.75,interpolation='nearest')
        q4 = data[col].quantile(1,interpolation='nearest')
        IQR = q3 -q1
        global LLP
        global ULP
        LLP = q1 - 1.5*IQR
        ULP = q3 + 1.5*IQR
        if data[col].min() > LLP and data[col].max() < ULP:
            print("No outliers in",i)
        else:
            print("There are outliers in",i)
            x = data[data[col]<LLP][col].size
            y = data[data[col]>ULP][col].size
            a.append(i)
            print('Count of outliers are:',x+y)
global a
a = []
for i in x:
    count_outliers(telecom_dataset,i)

#Cleaning and Transforming Data
#Data Processing
#On Hot Encoding
df1=pd.get_dummies(data=telecom_dataset,columns=['Gender', "Married", 'PhoneService', 'MultipleLines', 'InternetService', 'TechSupport', 'StreamingTV','StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'], drop_first=True)
print(df1.columns)
print(df1.head())
print(df1.shape)

churn_corr_matrix = df1.corr()
churn_corr_matrix["Churn_Yes"].sort_values(ascending = False).plot(kind='bar',figsize = (15,10),color="yellowgreen").set_title("BAR PLOT OF CORRELATION MATRIX BASED ON CHURN",fontsize=20,fontweight="bold")
plt.xlabel("Columns/Column's Features")
plt.ylabel("Correlation")
plt.show()


from sklearn.impute import SimpleImputer

# The imputer will replace missing values with the mean of the non-missing values for the respective columns

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

df1.TotalCharges = imputer.fit_transform(df1["TotalCharges"].values.reshape(-1, 1))
#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df1.drop(['Churn_Yes'],axis=1))
scaled_features = scaler.transform(df1.drop('Churn_Yes',axis=1))

#Feature Selection
from sklearn.model_selection import train_test_split
X = scaled_features
Y = df1['Churn_Yes']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=44)
#Prediction using Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)

predLR = logmodel.predict(X_test)

print(classification_report(Y_test, predLR))

#Calculate the classification report
report = classification_report(Y_test, predLR, target_names=['Churn_No', 'Churn_Yes'])

#Split the report into lines
lines = report.split('\n')

#Split each line into parts
parts = [line.split() for line in lines[2:-5]]

#Extract the metrics for each class
class_metrics = dict()
for part in parts:
    class_metrics[part[0]] = {'Precision': float(part[1]), 'Recall': float(part[2]), 'F1-score': float(part[3]), 'Support': int(part[4])}

#Create a bar chart for each metric
fig, ax = plt.subplots(1, 4, figsize=(12, 4))
metrics = ['Precision', 'Recall', 'F1-score', 'Support']
for i, metric in enumerate(metrics):
    ax[i].bar(class_metrics.keys(), [class_metrics[key][metric] for key in class_metrics.keys()])
    ax[i].set_title(metric)

#Display the plot
pl.suptitle("CLASS METRICS",fontweight ="bold",fontsize=20)
plt.show()

confusion_matrix_LR = confusion_matrix(Y_test, predLR)
#Create a heatmap of the matrix using matshow()

plt.matshow(confusion_matrix(Y_test, predLR))
plt.title("HEATMAP OF CONFUSION MATRIX",fontweight="bold",fontsize=20)

#Add labels for the x and y axes
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')

for i in range(2):
    for j in range(2):
        plt.text(j, i, confusion_matrix_LR[i, j], ha='center', va='center')


#Add custom labels for x and y ticks
plt.xticks([0, 1], ["Not Churned", "Churned"])
plt.yticks([0, 1], ["Not Churned", "Churned"])
plt.show()
print(logmodel.score(X_train, Y_train))
print(accuracy_score(Y_test, predLR))


