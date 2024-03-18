#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.metrics import make_scorer,f1_score,r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score,roc_curve
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.preprocessing import binarize, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# In[2]:


import warnings
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# Creating functions which are being used repeateadly.

# In[3]:


# define a function to print accuracy metrics
def print_accuracy_metrics(Input,Output):
  print("Recall:", recall_score(Input, Output))
  print("Log Loss:", log_loss(Input, Output))
  print("Precision:", precision_score(Input, Output))
  print("Accurcay:", accuracy_score(Input, Output))
  print("AUC: ", roc_auc_score(Input, Output))
  print("F1 Score:", f1_score(Input, Output))
  confusion_matrix_value = confusion_matrix(Input,Output)
  print('Confusion matrix:\n', confusion_matrix_value)
  class_names=[0,1] # name  of classes
  fig, ax = plt.subplots()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names)
  plt.yticks(tick_marks, class_names)
  # create heatmap
  sns.heatmap(pd.DataFrame( confusion_matrix_value), annot=True, cmap="YlGnBu" ,fmt='g')
  ax.xaxis.set_label_position("top")
  plt.tight_layout()
  plt.title('Confusion matrix', y=1.1)
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')


# In[4]:


# defined a function to print cross validation score
scoring = {'recall' : make_scorer(recall_score)}
def cross_validation_metrics(log_reg, X, y):
 log_reg_score = cross_val_score(log_reg, X,y,cv=5,scoring='recall')
 print('Logistic Regression Cross Validation Score(Recall): ', round(log_reg_score.mean() * 100, 2))


# In[5]:


# function to draw ROC curve
def plot_auc_curve(model,):
  auc = roc_auc_score(y, y_pred_prob)
  fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
  
  plt.plot(fpr, tpr)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.title('ROC Curve\n AUC={auc}'.format(auc = auc))
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.grid(True)


# In[6]:


### Reading data as a pandas dataframe
data = pd.read_csv('/Users/pooja/Downloads/creditcard.csv.zip')


# In[7]:


data.head()


# Observations

# In[8]:


data.describe()


# In[9]:


data.columns


# In[10]:


#### Checking for null values in dataset
data.isnull().sum().max()


# In[11]:


#### There are no null values in dataset 
####  Checking for unique values of ids
data.nunique()


# Data is pretty clean and there are no duplicate ids are present now let's check distribution of each feature.

# In[12]:


# Plot the histograms of each 
data.hist(bins=50, figsize=(30,20))
plt.show()


# We can observe that all the features in dataset are scaled except amount and time. So, in next step I am going to scale Amount column in dataset and delete time column.

# In[13]:


data['normal_amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Amount','Time'], axis=1)
X = data.loc[:,data.columns != 'Class']
y = data.loc[:,data.columns == 'Class']


# In[14]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))

bins = 30

ax1.hist(data.normal_amount[data.Class == 1], bins = bins)
ax1.set_title('Fraud')

ax2.hist(data.normal_amount[data.Class == 0], bins = bins)
ax2.set_title('Normal')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.yscale('log')
plt.show()


# In[15]:


# Now lets check the class distributions
sns.countplot(x="Class",data=data)
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[16]:


# Showing ratio
print("Percentage of normal transactions: ", len(data[data.Class == 0])/len(data))
print("Percentage of fraud transactions: ", len(data[data.Class == 1])/len(data))
print("Total number of transactions in data: ", len(data))


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)


# In[18]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[19]:


lr = LogisticRegression()
lr.fit(X_train,y_train)


# In[20]:


# Accuracy metrics for 
y_pred = lr.predict(X_test)
cross_validation_metrics(lr,X_train,y_train)
print_accuracy_metrics(y_test,y_pred)


# Observatios By observing the accuracy we can conclude that algorithm is performing extremely well . But it’s not true. As most of the labels 0, even random guess gives you 99% accuracy. So we need a better measure to understand the performance of the model.
# 
# Recall Recall is a measure which measures the ability of model to predict right for a given label. In our case, we want to test the model how accurately it can recall fraud cases as we are interested in that. As you can observe from the results, the recall for 1.0 is only 0.6016 compared to 99% for 0. So our model is not doing a good job of recognising frauds. So this shows that how imbalanced data is effecting accuracy of model.

# 2] Using Class Weight (Logistic regression) Scikit-learn logistic regression has a option named class_weight when specified does class imbalance handling implicitly. So trying to predict using this technique

# In[21]:


lr_balanced = LogisticRegression(class_weight = 'balanced')
lr_balanced.fit(X_train,y_train)


# In[22]:


y_balanced_pred = lr_balanced.predict(X_test)
cross_validation_metrics(lr_balanced,X_train,y_train)
print_accuracy_metrics(y_test,y_balanced_pred)


# In[23]:


y_balanced_pred_prob = lr_balanced.predict_proba(X_test)[:, 1]


# In[24]:


print('Prob:', y_balanced_pred_prob[0:20])


# In[25]:


print('Prob:', y_balanced_pred[0:20])


# Undersampling of the dataset Undersampling is one of the techniques used for handling class imbalance. In this technique, we under sample majority class to match the minority class. So in our example, we take random sample of non-fraud class to match number of fraud samples. This makes sure that the training data has equal amount of fraud and non-fraud samples.

# In[26]:


number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)


# In[27]:


normal_indices = data[data.Class == 0].index


# In[29]:


random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)


# In[30]:


under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])


# In[31]:


under_sample = data.iloc[under_sample_indices,:]


# In[32]:


under_sample.shape


# So there are total 984 observations in our undersample dataframe.
# 
# Visualising Undersampled Data.

# In[34]:


# Now lets check the class distributions
sns.countplot(x="Class",data=under_sample)
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")


# Splitting under sampled dataframe.

# In[35]:


X_under = under_sample.loc[:,under_sample.columns != 'Class']
y_under = under_sample.loc[:,under_sample.columns == 'Class']
X_under_train, X_under_test, y_under_train, y_under_test = train_test_split(X_under,y_under,test_size = 0.3, random_state = 0)


# 3] Logistic regression with C=0.01.

# In[36]:


# lr_under_C2 = LogisticRegression(C=0.1, penalty='l1', solver='liblinear')

lr_under_C1 = LogisticRegression(C=0.01,penalty = 'l1',solver='liblinear')
lr_under_C1.fit(X_under_train,y_under_train)


# In[37]:


#  Prediction on original dataframe
y_pred_full_model1 = lr_under_C1.predict(X_test)
cross_validation_metrics(lr_under_C1,X_under_train,y_under_train)
print_accuracy_metrics(y_test,y_pred_full_model1)


# 3] Logistic regression with C=0.1.

# In[38]:


lr_under_C2 = LogisticRegression(C=0.1,penalty = 'l1',solver='liblinear')
lr_under_C2.fit(X_under_train,y_under_train)


# In[39]:


# Prediction on original dataset
y_pred_full_model2 = lr_under_C2.predict(X_test)
cross_validation_metrics(lr_under_C2,X_under_train,y_under_train)
print_accuracy_metrics(y_test,y_pred_full_model2)


# 3] Logistic regression with C=1.

# In[40]:


lr_under_C3 = LogisticRegression(C=1,penalty = 'l1',solver='liblinear')
lr_under_C3.fit(X_under_train,y_under_train)


# In[41]:


# Prediction on original dataset
y_pred_full_model3 = lr_under_C3.predict(X_test)
cross_validation_metrics(lr_under_C3,X_under_train,y_under_train)
print_accuracy_metrics(y_test,y_pred_full_model3)


# 3] Logistic regression with C=10.

# In[42]:


lr_under_C4 = LogisticRegression(C=10,penalty = 'l1',solver='liblinear')
lr_under_C4.fit(X_under_train,y_under_train)


# In[43]:


# Prediction on original dataset
y_pred_full_model4 = lr_under_C4.predict(X_test)
cross_validation_metrics(lr_under_C4,X_under_train,y_under_train)
print_accuracy_metrics(y_test,y_pred_full_model4)


# 7] Decision Tree Classifier.
# 

# In[44]:


DecisionTreeClassifier= DecisionTreeClassifier()
DecisionTreeClassifier.fit(X_under_train,y_under_train)


# In[45]:


# Prediction on original dataset
y_pred_DecisionTree = DecisionTreeClassifier.predict(X_test)
cross_validation_metrics(DecisionTreeClassifier,X_under_train,y_under_train)
print_accuracy_metrics(y_test,y_pred_DecisionTree)


# In[46]:


plt.figure(0).clf()

fpr, tpr, thresh = metrics.roc_curve(y_test, y_balanced_pred)
auc = metrics.roc_auc_score(y_test, y_balanced_pred)
plt.plot(fpr,tpr,label="Logistic Regtession Class weight, auc="+ '{0:.3f}'.format(auc))

fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred_full_model1)
auc = metrics.roc_auc_score(y_test, y_pred_full_model1)
plt.plot(fpr,tpr,label="Logistic regression(C=0.01), auc="+ '{0:.3f}'.format(auc))

fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred_full_model2)
auc = metrics.roc_auc_score(y_test, y_pred_full_model2)
plt.plot(fpr,tpr,label="Logistic regression(C=0.1), auc="+'{0:.3f}'.format(auc))

fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred_full_model3)
auc = metrics.roc_auc_score(y_test, y_pred_full_model3)
plt.plot(fpr,tpr,label="Logistic regression(C=1), auc="+'{0:.3f}'.format(auc))

fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred_full_model4)
auc = metrics.roc_auc_score(y_test, y_pred_full_model4)
plt.plot(fpr,tpr,label="Logistic regression(C=10), auc="+'{0:.3f}'.format(auc))

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC Curve\n AUC={auc}'.format(auc = auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.legend(loc="lower right")


# 
# Obviously, trying to increase recall, tends to come with a decrease of precision. However, in our case, if we predict that a transaction is fraudulent and turns out not to be, is not a massive problem compared to the opposite.

# In[ ]:




