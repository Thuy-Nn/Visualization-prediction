import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

# 1. Load data


# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

# metadata
print(adult.metadata)

# variable information
print(adult.variables)

print(adult.data.keys())

print(adult.data.original)

# 2. Data understanding
df = adult.data.original
df.isna().sum()

df.head(200)

df.shape

df.columns

df.dtypes

#Chart 'Race Distribution'
race_counts = df["race"].value_counts()

sns.barplot(x=race_counts.index, y=race_counts.values)

plt.xticks(rotation=90)
plt.xlabel('Race')
plt.ylabel('Count')
plt.title('Race Distribution')
plt.show()

#Chart 'Hours-per-week Distribution'
plt.hist(df["hours-per-week"], edgecolor='black')
plt.xlabel('Hours-per-week')
plt.ylabel('Frequency')
plt.title('Hours-per-week Distribution')
plt.show()

#Chart 'Capital-loss Distribution'
plt.hist(df["capital-loss"], edgecolor='black')
plt.xlabel('Capital-loss')
plt.ylabel('Frequency')
plt.title('Capital-loss Distribution')
plt.show()

#Chart 'Capital-gain Distribution '
plt.hist(df["capital-gain"], edgecolor='black')
plt.xlabel('Capital-gain')
plt.ylabel('Frequency')
plt.title('Capital-gain Distribution')
plt.show()

#Chart 'Age Distribution'
plt.hist(df["age"], bins=10, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()

#Chart 'Education-num Distribution'
plt.hist(df['education-num'], bins=10, edgecolor='black')
plt.xlabel('Education-num')
plt.ylabel('Frequency')
plt.title('Education-num Distribution')
plt.show()

#Chart 'Native-country Distribution'
country_counts = df["native-country"].value_counts()

sns.barplot(x=country_counts.index, y=country_counts.values)

plt.xticks(rotation=90)
plt.xlabel('Native-country')
plt.ylabel('Count')
plt.title('Native-country Distribution')
plt.show()

#Chart 'gender Distribution'
gender_counts = df["sex"].value_counts()

sns.barplot(x=gender_counts.index, y=gender_counts.values)

plt.xticks(rotation=90)
plt.xlabel('gender')
plt.ylabel('Count')
plt.title('gender Distribution')
plt.show()

#Chart 'Relationship Distribution'
relationship_counts = df['relationship'].value_counts()

sns.barplot(x= relationship_counts.index, y=relationship_counts.values)

plt.xticks(rotation = 90)
plt.xlabel('Relationship')
plt.ylabel('Count')
plt.title('Relationship Distribution')
plt.show()

#Chart 'Marital-status Distribution'
m_status_counts = df["marital-status"].value_counts()

sns.barplot(x=m_status_counts.index, y=m_status_counts.values)

plt.xticks(rotation=90)
plt.xlabel('marital-status')
plt.ylabel('Count')
plt.title('marital-status Distribution')
plt.show()

#Chart 'education distribution'
education_counts = df["education"].value_counts()

sns.barplot(x=education_counts.index, y=education_counts.values)

plt.xticks(rotation=90)
plt.xlabel('education')
plt.ylabel('Count')
plt.title('education Distribution')
plt.show()

#chart 'Workclass'
workclass_counts = df["workclass"].value_counts()

sns.barplot(x=workclass_counts.index, y=workclass_counts.values)

plt.xticks(rotation=90)
plt.xlabel('Workclass')
plt.ylabel('Count')
plt.title('Workclass Distribution')
plt.show()

#Chart 'Occupation Distribution'
occupation_counts = df['occupation'].value_counts()

sns.barplot(x=occupation_counts.index, y=occupation_counts.values)

plt.xticks(rotation=90)
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.title('Occupation Distribution')
plt.show()

#Chart 'Income'
income_counts = df["income"].value_counts()

sns.barplot(x=income_counts.index, y=income_counts.values)

plt.xticks(rotation=90)
plt.xlabel('income')
plt.ylabel('Count')
plt.title('income Distribution')
plt.show()

# 3. Data Cleaning

#check value unique
df.nunique()

#replace '?'
df['workclass'] = df['workclass'].apply(lambda x: 'Private' if x == '?' else x)

df['occupation'] = df['occupation'].apply(lambda x: 'Prof-specialty' if x == '?' else x)

df["native-country"] = df["native-country"].apply(lambda x: "United-States" if x == "?" else x)

#remove '.' from column 'income'
df['income'] = df['income'].str.replace('.', '')
print(df['income'].value_counts().index)

#check missing values
missing_values = df.isnull().sum()
print(missing_values)

# 4. Feature Engineering

#Delete column 'education', because it has the same meaning with column 'education-num'
df=df.drop('education', axis=1)

'''
Label Encoding for Columns 
'''
label_encoder = LabelEncoder()
df['workclass'] = label_encoder.fit_transform(df['workclass'])
print(df['workclass'])

label_encoder = LabelEncoder()
df['relationship'] = label_encoder.fit_transform(df['relationship'])
print(df['relationship'])

label_encoder = LabelEncoder()
df['race'] = label_encoder.fit_transform(df['race'])
print(df['race'])

label_encoder = LabelEncoder()
df['sex'] = label_encoder.fit_transform(df['sex'])
print(df['sex'])

label_encoder = LabelEncoder()
df['native-country'] = label_encoder.fit_transform(df['native-country'])
print(df['native-country'])

label_encoder = LabelEncoder()
df['marital-status'] = label_encoder.fit_transform(df['marital-status'])
print(df['marital-status'])

label_encoder = LabelEncoder()
df['occupation'] = label_encoder.fit_transform(df['occupation'])
print(df['occupation'])

df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
print(df['income'])

df.head()
# 5. Modeling

correlation_matrix = df.corr(method='pearson')
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('correlation matrix')
plt.show()

occupation_income_ratios = df.groupby('occupation')['income'].value_counts(normalize=True).unstack()
occupation_income_ratios.columns = ['Income <=50K (%)', 'Income >50K (%)']
occupation_income_ratios.sort_values(by='Income <=50K (%)', ascending=False)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

features = df.drop(['income', 'workclass', 'marital-status', 'relationship', 'fnlwgt',
                    'occupation', 'race', 'capital-loss', 'native-country',
                    # 'age', 'sex', 'capital-gain', 'hours-per-week'
                    ], axis =1)
target = df['income']

X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
logistic_model = LogisticRegression(max_iter=500, random_state=42)
logistic_model.fit(X_train, y_train)

# Predictions on the validation set
y_pred = logistic_model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)
class_report = classification_report(y_val, y_pred)

print(accuracy)
print(conf_matrix)
print(class_report)

#Model GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier

clf_model = GradientBoostingClassifier()
clf_model.fit(X_train, y_train)

y_pred = clf_model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)
class_report = classification_report(y_val, y_pred)

print(accuracy)
print(conf_matrix)
print(class_report)

#Model AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

clf_model = AdaBoostClassifier()
clf_model.fit(X_train, y_train)

y_pred = clf_model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)
class_report = classification_report(y_val, y_pred)

print(accuracy)
print(conf_matrix)
print(class_report)
'''
**Feature Selection:** 

**Data Splitting:** 

**Model Training:**

**Model Evaluation:** 
'''
