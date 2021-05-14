#!/usr/bin/env python
# coding: utf-8

# # Performance Prediction Challenge

# In[ ]:


import pandas as pd
import numpy as np 
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import preprocessing


# In[ ]:


from google.colab import drive
drive.mount('/content/drive/')


# In[ ]:


data_train=pd.DataFrame(np.loadtxt('/content/drive/My Drive/ada_train.data'))
data_test=pd.DataFrame(np.loadtxt('/content/drive/My Drive/ada_test.data'))
labels_train=pd.DataFrame(np.loadtxt('/content/drive/My Drive/ada_train.labels'))
data_valid=pd.DataFrame(np.loadtxt('/content/drive/My Drive/ada_valid.data'))
labels_valid=pd.DataFrame(np.loadtxt('/content/drive/My Drive/ada_valid.labels'))


# # 1. Exploratory data analysis #

# In[ ]:


data_train.head(5)


# In[ ]:


data_train.shape


# We have 49 features, all numerical values.  
# The training set contains 4147 examples.  
# We see that the last column has missing values. Let's see how many missing values each column has.

# In[ ]:


# we are reshaping it just to make the output more aesthetical (otherwise it would look like a long column that one has to scroll through)
data_train.isnull().sum().values.reshape(1,-1)


# Let's run some descriptive statistics on our data.

# In[ ]:


data_train.describe() 


# Most of our features have values between 0 and 1.   
# Only 6 of them have values up to 1000.

# In[ ]:


data_train.max().values.reshape(1,-1)


# By the look of the descriptive statistics above, it looks like some of the features might be binary. Let's count the **unique values per column**.

# In[ ]:


data_train.nunique().values.reshape(1,-1)


# Indeed, the most of our features are binary, as suspected.  
# Let's get the indexes for those columns.

# In[ ]:


not_binary_columns = data_train.columns[data_train.nunique()>2].values
print(not_binary_columns)


# There are also two columns with a single value.  

# In[ ]:


one_val_columns = data_train.columns[data_train.nunique()==1].values
print(one_val_columns)


# Let's have a look at histograms to see the distribution of each features.

# In[ ]:


data_train.hist(figsize=(20,20))


# Let's have a look at histograms to see the distribution of each features with the max value >1.

# In[ ]:


max_values=(data_train.max()).tolist()
max=pd.DataFrame()
for i in range(len(data_train.columns)):
    if(max_values[i]>1):
        a=data_train[[i]]
        max=pd.concat([max,a],axis=1)
max.hist(figsize=(20,12))


# Let's see them as boxplots, so we can get a better feel of the outliers.

# In[ ]:


plt.figure(figsize=(15,8))
sns.boxplot(data=max)


# Observations:  
# 1. There are too many outliers in columns 3, 14, 24 and 39  
# 2. Columns 9 and 31 have almost only 0 values

# In[ ]:


plt.figure(figsize=(15,8))
plt.title('Column 9 versus label')
plt.scatter(data_train[9],labels_train)


# In[ ]:


fig, axs = plt.subplots(1, 2, constrained_layout=True)
fig.suptitle('Column 9 values versus labels', fontsize=16)
fig.set_figheight(4)
fig.set_figwidth(12)

axs[0].set_title('Class distriutio for 0 values in column 9')
axs[0].bar([-1,1],labels_train[data_train[9]==0].value_counts().values)

axs[1].set_title('Class distriutio for non-0 values in column 9')
axs[1].bar([-1,1],labels_train[data_train[9]>0].value_counts().values)


# In[ ]:


plt.figure(figsize=(15,8))
plt.title('Column 31 versus label')
plt.scatter(data_train[31],labels_train)


# In[ ]:


fig, axs = plt.subplots(1, 2, constrained_layout=True)
fig.suptitle('Column 31 values versus labels', fontsize=16)
fig.set_figheight(4)
fig.set_figwidth(12)

axs[0].set_title('Class distriutio for 0 values in column 31')
axs[0].bar([-1,1],labels_train[data_train[31]==0].value_counts().values)

axs[1].set_title('Class distriutio for non-0 values in column 31')
axs[1].bar([-1,1],labels_train[data_train[31]>0].value_counts().values)


# Distribution of classes in our training set  
# 
# 3000 class -1  
# 1000 class 1   

# In[ ]:


plt.title('Distribution of classes in our training set')
plt.bar([-1, 1], labels_train.value_counts())


# ### Clustering Analysis - Outliers detection

# In[ ]:


from sklearn.cluster import DBSCAN  
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 


# In[ ]:


get_ipython().system('wget -O ADA.zip http://clopinet.com/isabelle/Projects/modelselect/datasets/ADA.zip')


# In[ ]:


get_ipython().system('rm -r ADA')
get_ipython().system('unzip ADA.zip')


# In[ ]:


ada_train_labels = pd.DataFrame(pd.read_table("ADA/ada_train.labels").values)
ada_valid_labels = pd.DataFrame(pd.read_table("ADA/ada_valid.labels").values)
ada_train_labels.head()


# In[ ]:


ada_train_labels.shape


# In[ ]:


ada_train = pd.DataFrame(pd.read_table("ADA/ada_train.data", sep=" ", ).values)
ada_valid = pd.DataFrame(pd.read_table("ADA/ada_valid.data", sep=" ", ).values)
ada_train.head()


# In[ ]:


ada_train.shape


# In[ ]:


ada_train = ada_train.drop(48, axis=1)
ada_valid = ada_valid.drop(48, axis=1)


# In[ ]:


epsilon = 0.3
minimumSamples = 7
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(ada_train)
labels = db.labels_
labels


# In[ ]:


# Firts, create an array of booleans using the labels from db.
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask


# In[ ]:


# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters_


# In[ ]:


# Remove repetition in labels by turning it into a set.
unique_labels = set(labels)
unique_labels


# In[ ]:


# Create colors for the clusters.
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))


# In[ ]:


# Plot the points with colors
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    # Plot the datapoints that are clustered
    xy = ada_train[class_member_mask & core_samples_mask]
    plt.scatter(xy.values[:, 3], xy.values[:, 14],s=50, c=[col], marker=u'o', alpha=0.5)

    # Plot the outliers
    xy = ada_train[class_member_mask & ~core_samples_mask]
    plt.scatter(xy.values[:, 3], xy.values[:, 14],s=50, c=[col], marker=u'o', alpha=0.5)


# In[ ]:


from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 


# In[ ]:


agglom = AgglomerativeClustering(n_clusters = 2, linkage = 'average')


# In[ ]:


agglom.fit(ada_train.values, ada_train_labels.values)


# In[ ]:


dist_matrix = distance_matrix(ada_train.values, ada_train.values) 
print(dist_matrix)


# In[ ]:


Z = hierarchy.linkage(dist_matrix, 'complete')


# In[ ]:


#dendro = hierarchy.dendrogram(Z)


# In[ ]:


sns.displot(ada_train[14], kind="kde")


# In[ ]:


not_binary_columns = ada_train.columns[ada_train.nunique()>2].values
print(not_binary_columns)


# In[ ]:


sns.displot(ada_train[[3, 9, 14, 24, 31, 39]], kind="kde")


# In[ ]:


sns.set_theme(style="ticks")
sns.pairplot(ada_train[[3, 9, 14, 24, 31, 39]])


# In[ ]:


plt.subplots(figsize=(8, 5))
sns.heatmap(ada_train[[3, 9, 14, 24, 31, 39]].corr(), annot=True, cmap="RdYlGn")
plt.show()


# In[ ]:


#ada_train.corr()


# # 2. Data engineering #

# ## Training set

# Let's drop the features that contain only 0s.

# In[ ]:


data_train=data_train.drop(13, axis=1)
data_train=data_train.drop(20, axis=1)


# Dealing with outliers: 
# 1. in columns [3 14 24 39] we have too many values that qualify as outlier. So we will let them be
# 2. in columns 9 and 31 we have 90% of values equal to 0 and the rest go up to 1000. We don't see a correlation with class value. So we drop these columns completely.

# In[ ]:


data_train=data_train.drop(9, axis=1)
data_train=data_train.drop(31, axis=1)


# Normalizing the 4 remaining features that have values between 0 and 1000

# In[ ]:


from sklearn.preprocessing import StandardScaler

data_train_scaled = data_train.copy()
col_names = [3, 14, 24, 39]
features = data_train_scaled[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
data_train_scaled[col_names] = features
data_train_scaled.head(3)


# In[ ]:


from sklearn import preprocessing
data_train_normal = data_train.copy()
col_names = [3, 14, 24, 39]
features = data_train_normal[col_names]
features = preprocessing.normalize(features)
data_train_normal[col_names] = features
data_train_normal.head(3)


# ##Validation set

# In[ ]:


data_valid.describe()


# In[ ]:


#dropping the varibles with 0 values
data_valid=data_valid.drop(13, axis=1)
data_valid=data_valid.drop(20, axis=1)

#data_valid=data_valid.drop(46, axis=1) 


# In[ ]:


#dropping the varibles with 0 values 
data_valid=data_valid.drop(9, axis=1)
data_valid=data_valid.drop(31, axis=1)


# In[ ]:


data_valid_normal = data_valid.copy()
col_names = [3, 14, 24, 39]
features = data_valid_normal[col_names]
features = preprocessing.normalize(features)
data_valid_normal[col_names] = features
data_valid_normal.head(3)


# # 3.Models building

# ### Find best k for kNN

# In[ ]:


from sklearn.neighbors import NearestNeighbors


# In[ ]:


N = 50
train_mcc = []
test_mcc = []
for k in range(1,N):
    neigh = KNeighborsClassifier(n_neighbors = k).fit(data_train_normal.values,labels_train)
    y_hat = neigh.predict(data_valid_normal)
    train_mcc.append(metrics.matthews_corrcoef(labels_train, neigh.predict(data_train_normal.values)))
    test_mcc.append(metrics.matthews_corrcoef(labels_valid, y_hat))

plt.plot(range(1,N), train_mcc, label='train')
plt.plot(range(1,N), test_mcc, label='test')
plt.legend(loc='best')
plt.xlabel('Value of k')
plt.ylabel('MCC of model')
plt.tight_layout()
plt.show()


# In[ ]:


Ks = 30
mean_mcc = np.zeros((Ks-1, 4))
std_mcc = np.zeros((Ks-1, 4))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    auto_model = KNeighborsClassifier(n_neighbors = n).fit(data_train_normal.values,labels_train.values)
    ball_model = KNeighborsClassifier(n_neighbors = n, algorithm='ball_tree').fit(data_train_normal.values,labels_train.values)
    kd_model = KNeighborsClassifier(n_neighbors = n, algorithm='kd_tree').fit(data_train_normal.values,labels_train.values)
    brute_model = KNeighborsClassifier(n_neighbors = n, algorithm='brute').fit(data_train_normal.values,labels_train.values)
    yhat=auto_model.predict(data_valid_normal.values)
    mean_mcc[n-1, 0] = metrics.matthews_corrcoef(labels_valid.values, yhat)
    std_mcc[n-1, 0]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
    yhat=ball_model.predict(data_valid_normal.values)
    mean_mcc[n-1, 1] = metrics.matthews_corrcoef(labels_valid.values, yhat)
    std_mcc[n-1, 1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
    yhat=kd_model.predict(data_valid_normal.values)
    mean_mcc[n-1, 2] = metrics.matthews_corrcoef(labels_valid.values, yhat)
    std_mcc[n-1, 2]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
    yhat=brute_model.predict(data_valid_normal.values)
    mean_mcc[n-1, 3] = metrics.matthews_corrcoef(labels_valid.values, yhat)
    std_mcc[n-1, 3]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_mcc


# In[ ]:


plt.plot(range(1,Ks),mean_mcc[:, 0],'g')
plt.fill_between(range(1,Ks),mean_mcc[:, 0] - 1 * std_mcc[:, 0],mean_mcc[:, 0] + 1 * std_mcc[:, 0], alpha=0.10)
plt.legend(('MCC ', '+/- std'))
plt.ylabel('MCC for auto')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


# In[ ]:


print( "The best mcc for auto model was", mean_mcc[:,0].max(), "with k=", mean_mcc[:,0].argmax()+1) 


# In[ ]:


plt.plot(range(1,Ks),mean_mcc[:, 1],'g')
plt.fill_between(range(1,Ks),mean_mcc[:, 1] - 1 * std_mcc[:, 1],mean_mcc[:, 1] + 1 * std_mcc[:, 1], alpha=0.10)
plt.legend(('MCC ', '+/- std'))
plt.ylabel('MCC for ball')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


# In[ ]:


print( "The best mcc for auto ball was", mean_mcc[:,1].max(), "with k=", mean_mcc[:,1].argmax()+1) 


# In[ ]:


plt.plot(range(1,Ks),mean_mcc[:, 2],'g')
plt.fill_between(range(1,Ks),mean_mcc[:, 2] - 1 * std_mcc[:, 2],mean_mcc[:, 2] + 1 * std_mcc[:, 2], alpha=0.10)
plt.legend(('MCC ', '+/- std'))
plt.ylabel('MCC for kd')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


# In[ ]:


print( "The best mcc for auto kd was", mean_mcc[:,2].max(), "with k=", mean_mcc[:,2].argmax()+1) 


# In[ ]:


neigh = KNeighborsClassifier(n_neighbors = 21).fit(data_train_normal.values,labels_train.values)
metrics.plot_roc_curve(neigh, data_valid_normal.values, labels_valid.values)
plt.show()


# In[ ]:


metrics.plot_precision_recall_curve(neigh, data_valid_normal.values, labels_valid.values)
plt.show()


# In[ ]:


metrics.plot_confusion_matrix(neigh, data_valid_normal.values, labels_valid.values)
plt.show()


# In[ ]:


yhat=neigh.predict(data_valid_normal.values)
fpr, tpr, thresholds = metrics.roc_curve(labels_valid.values, yhat)
metrics.auc(fpr, tpr)


# In[ ]:


metrics.balanced_accuracy_score(labels_valid.values, yhat)


# In[ ]:


from sklearn.inspection import permutation_importance
result = permutation_importance(neigh, data_valid_normal.values, yhat, n_repeats=10, random_state=0)
result


# In[ ]:


indices = np.argsort(result.importances_mean)[::-1]
indices


# In[ ]:


result.importances_mean[indices]


# In[ ]:


result.importances_std[indices]


# In[ ]:


plt.figure()
plt.title("Feature importances")
plt.bar(range(data_valid_normal.shape[1]), result.importances_mean[indices],
        color="r", yerr=result.importances_std[indices], align="center")
plt.xticks(range(data_valid_normal.shape[1]), indices)
plt.xlim([-1, data_valid_normal.shape[1]])
plt.show()


# In[ ]:


#Not finished yet - Gabriel

print( "Features that does not have importance for the model", [i for i n result.importances_mean[indices]]) 


# ## 3.0 Feature Selection ##
# 
# RFE for selecting the best features on which to train our models

# In[ ]:


import time
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE

k_neigh = 14

X = data_train_normal.copy()
y = labels_train.copy()

# get a list of LogisticRegression models to evaluate
def get_lr_models():
	models = dict()
	for i in range(20, 30):
		rfe = RFE(estimator=(LogisticRegression()), n_features_to_select=i)
		model = LogisticRegression()
		models[str(i)] = Pipeline(steps=[('s',rfe),('m',model)])
	return models

# get a list of kNN models to evaluate
def get_knn_models():
	models = dict()
	for i in range(20, 30):
		rfe = RFE(estimator=KNeighborsClassifier(n_neighbors = k_neigh), n_features_to_select=i)
		model = KNeighborsClassifier(n_neighbors = k_neigh)
		models[str(i)] = Pipeline(steps=[('s',rfe),('m',model)])
	return models

# get a list of LDA models to evaluate
def get_lda_models():
	models = dict()
	for i in range(20, 30):
		rfe = RFE(estimator=LinearDiscriminantAnalysis(), n_features_to_select=i)
		model = LinearDiscriminantAnalysis()
		models[str(i)] = Pipeline(steps=[('s',rfe),('m',model)])
	return models

# get a list of Decision Tree models to evaluate
def get_tree_models():
	models = dict()
	for i in range(20, 30):
		#we haven't found the optimum alpha yet. It's in the code below, in the Decision Tree section
		#rfe = RFE(estimator=DecisionTreeClassifier(ccp_alpha = optimum_alpha), n_features_to_select=i)
		#model = DecisionTreeClassifier(ccp_alpha = optimum_alpha)
		
		rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=i)
		model = DecisionTreeClassifier()
		models[str(i)] = Pipeline(steps=[('s',rfe),('m',model)])
	return models

# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores

def evaluate_all_models(models, title):
  start = time.time()

  # evaluate the models and store results
  results, names = list(), list()
  for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
  # plot model performance for comparison
  plt.boxplot(results, labels=names, showmeans=True)
  plt.title(title)
  plt.ylim((0.5,1))
  plt.show()

  stop = time.time()

  print(f'Total time (seconds): {stop-start}')


# In[ ]:


evaluate_all_models(get_lr_models(), 'Logistic regression accuracy versus number of features for training')


# In[ ]:


evaluate_all_models(get_lda_models(), 'LDA accuracy versus number of features for training')


# In[ ]:


evaluate_all_models(get_tree_models(), 'Decistion Tree accuracy versus number of features for training')


# For kNN we cannot do the same operations as above. It needs a tweak because 'The classifier does not expose "coef_" or "feature_importances_" attributes'

# ##3.1 Naive bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB

def Average(lst): 
    return sum(lst) / len(lst) 


# In[ ]:


acc_list = []
mcc_list = []
for i in range(1, len(ada_train.columns.values) + 1):
  acc_l = []
  mcc_l = []
  for j in range(10):
    X_train, X_val, Y_train, Y_val = train_test_split(X[:,:i], y, test_size=0.2, random_state=12)
    #building the model
    model = GaussianNB()
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_val)
    acc_l.append(metrics.accuracy_score(Y_val, y_pred))
    mcc_l.append(metrics.matthews_corrcoef(Y_val, y_pred))
  acc_l.remove(max(acc_l))
  acc_l.remove(min(acc_l))
  acc = Average(acc_l)
  acc_list.append(acc)
  mcc_l.remove(max(mcc_l))
  mcc_l.remove(min(mcc_l))
  mcc = Average(mcc_l)
  mcc_list.append(mcc)


# In[ ]:


plt.plot(ada_train.columns.values, acc_list, label='acc')
plt.plot(ada_train.columns.values, mcc_list, label='mcc')
plt.legend(loc='best')
plt.xlabel('Training Data (%)')
plt.ylabel('Score of the model')
plt.tight_layout()
plt.show()


# In[ ]:


max(acc_list)


# In[ ]:


max(mcc_list)


# In[ ]:


ada_y_valid = pd.DataFrame(pd.read_table("ADA/ada_valid.labels", sep=" ", ).values)
ada_y_valid.head()


# In[ ]:


ada_valid = pd.DataFrame(pd.read_table("ADA/ada_valid.data", sep=" ", ).values)
ada_valid.head()


# In[ ]:


ada_valid = ada_valid.drop(48, axis=1)


# In[ ]:


X_valid = preprocessing.normalize(ada_valid)
X_valid[:1]


# In[ ]:


X = preprocessing.normalize(ada_train)
X[:1]


# In[ ]:


y = ada_train_labels.values
y[:5]


# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

NB_model = GaussianNB()

#Train the model using the training sets
NB_model.fit(X, y)


# In[ ]:


#checking the accuracy score for validation dataset
y_pred = NB_model.predict(X_valid)

accuracy_score(y_pred,ada_y_valid.values)


# In[ ]:


metrics.plot_roc_curve(NB_model, X_valid, ada_y_valid.values)
plt.show()


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(ada_y_valid.values, y_pred, labels=[-1,1])
#np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['-1','1'],normalize= False,  title='Confusion matrix')


# In[ ]:


metrics.plot_precision_recall_curve(NB_model, X_valid, ada_y_valid.values)
plt.show()


# ## 3.2 Decision Trees ##

# ### 3.2.1. Explore results with default parameters ###

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(data_train_normal, labels_train, test_size=0.3, random_state=101)
dtree = DecisionTreeClassifier()

print(dtree.fit(X_train, y_train))

train_accuracy = dtree.score(X_train, y_train)

predictions = dtree.predict(X_test)
print('Training accuracy:', train_accuracy)

print('\nTesting')
print('Confusion matrix:\n', confusion_matrix(y_test,predictions))
print('\n')
report = classification_report(y_test,predictions)
print(classification_report(y_test,predictions))


# ## 3.2.2. Finding the optimum depth ##

# In[ ]:


from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

def max_depth_swipe(global_max_depth = 15, folds = 5):
  '''
  retain the accuracy for test and train in a 3D array
  lines:
    0 - train
    1 - test
  columns:
    iteration through kfolds (the number of folds is set in the variable above)
  depth of array:
    from min to max depth of our Trees  
  '''
  accuracy = np.zeros((2, folds, global_max_depth), dtype=float)

  #we are only interested in depth 3 -> 14 (incl.)
  for d in range(3,15):
    # create the tree model and set max depth
    kfoldtree = DecisionTreeClassifier(max_depth=d)

    cv = KFold(folds, random_state=1, shuffle=True)

    '''
    for scoring, we select 'accuracy'
    there are many other options for the scoring parameter, detailed here: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    '''
    cv_results = cross_validate(kfoldtree, data_train_normal, labels_train, scoring='accuracy', cv=cv, return_train_score=True)

    '''
    cv_results contains: fit_time, score_time, test_score and train_score. 
    we are only interested in the last two
    '''
    accuracy[0,:,d] = cv_results['train_score']
    accuracy[1,:,d] = cv_results['test_score']

  accuracy_mean = np.mean(accuracy, axis=1)
  accuracy_std = np.std(accuracy, axis=1)

  fig = plt.figure(figsize=(10,8))
  x = np.arange(global_max_depth)
  plt.errorbar(x[3:], accuracy_mean[0,3:], accuracy_std[0,3:], marker='^', capsize=5, label='train')
  plt.errorbar(x[3:], accuracy_mean[1,3:], accuracy_std[1,3:], marker='x', capsize=5, label='test')
  plt.legend(fontsize=14)

  plt.xlabel('Maximum depth param for the decision tree tree', fontsize=14)
  plt.xlim([2.5,15])

  plt.ylabel(f'Mean accuracy ({folds}-fold cross validation)', fontsize=14)
  plt.ylim([np.min(accuracy_mean[:,3:])-0.1,1])

  plt.title(f'Mean accuracy versus max tree depth ({folds}-fold cross-validation)', fontsize=18)

  plt.show()


# In[ ]:


#Train decision trees with max_depth up to 15, using 10-fold cross-validation
max_depth_swipe(15,10)


# **The optimal tree depth is between 3 and 5.**

# In[ ]:


optimum_depth = 5


# ## 3.2.3. Finding the optimum min_samples_split ##

# In[ ]:


from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

folds = 10
min_samples_splits = np.linspace(0.001, 1, 100, endpoint=True)

'''
retain the accuracy for test and train in a 3D array
lines:
  0 - train
  1 - test
columns:
  iteration through kfolds (the number of folds is set in the variable above)
depth of array:
  from min to max depth of our Trees  
'''
accuracy = np.zeros((2, folds, min_samples_splits.shape[0]), dtype=float)

i=0
#we are only interested in depth 3 -> 14 (incl.)
for min_samples_split in min_samples_splits:
  # create the tree model and set max depth
  kfoldtree = DecisionTreeClassifier(min_samples_split=min_samples_split, max_depth=5)

  cv = KFold(folds, random_state=1, shuffle=True)

  cv_results = cross_validate(kfoldtree, data_train_normal, labels_train, scoring='accuracy', cv=cv, return_train_score=True)

  '''
  cv_results contains: fit_time, score_time, test_score and train_score. 
  we are only interested in the last two
  '''
  accuracy[0,:,i] = cv_results['train_score']
  accuracy[1,:,i] = cv_results['test_score']
  i+=1

accuracy_mean = np.mean(accuracy, axis=1)
accuracy_std = np.std(accuracy, axis=1)


fig = plt.figure(figsize=(10,8))

plt.plot(min_samples_splits, accuracy_mean[0,:], '-', label='train')
plt.fill_between(min_samples_splits, accuracy_mean[0,:]-accuracy_std[0,:], accuracy_mean[0,:]+accuracy_std[0,:], alpha = 0.2)

plt.plot(min_samples_splits, accuracy_mean[1,:], '-',  label='test')
plt.fill_between(min_samples_splits, accuracy_mean[1,:]-accuracy_std[1,:], accuracy_mean[1,:]+accuracy_std[1,:], alpha = 0.2)

plt.legend(fontsize=14)

plt.xlabel('Min samples split', fontsize=14)

plt.ylabel(f'Mean accuracy ({folds}-fold cross validation)', fontsize=14)
plt.ylim([0.7,1])

plt.title(f'Mean accuracy versus min sample split ({folds}-fold cross-validation)', fontsize=18)

plt.show()


# ## 3.2.4. Optimum min samples leaf ##

# In[ ]:


folds = 5
min_samples_leaves = np.linspace(0.001, 0.5, 100, endpoint=True)

'''
retain the accuracy for test and train in a 3D array
lines:
  0 - train
  1 - test
columns:
  iteration through kfolds (the number of folds is set in the variable above)
depth of array:
  from min to max depth of our Trees  
'''
accuracy = np.zeros((2, folds, min_samples_splits.shape[0]), dtype=float)

i=0
#we are only interested in depth 3 -> 14 (incl.)
for min_samples_leaf in min_samples_leaves:
  # create the tree model and set max depth
  kfoldtree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, max_depth=4)

  cv = KFold(folds, random_state=1, shuffle=True)

  cv_results = cross_validate(kfoldtree, data_train_normal, labels_train, scoring='accuracy', cv=cv, return_train_score=True)

  '''
  cv_results contains: fit_time, score_time, test_score and train_score. 
  we are only interested in the last two
  '''
  accuracy[0,:,i] = cv_results['train_score']
  accuracy[1,:,i] = cv_results['test_score']
  i+=1

accuracy_mean = np.mean(accuracy, axis=1)
accuracy_std = np.std(accuracy, axis=1)


fig = plt.figure(figsize=(10,8))

plt.plot(min_samples_splits, accuracy_mean[0,:], '-', label='train')
plt.fill_between(min_samples_splits, accuracy_mean[0,:]-accuracy_std[0,:], accuracy_mean[0,:]+accuracy_std[0,:], alpha = 0.2)

plt.plot(min_samples_splits, accuracy_mean[1,:], '-',  label='test')
plt.fill_between(min_samples_splits, accuracy_mean[1,:]-accuracy_std[1,:], accuracy_mean[1,:]+accuracy_std[1,:], alpha = 0.2)

plt.legend(fontsize=14)

plt.xlabel('Min samples leaf', fontsize=14)

plt.ylabel(f'Mean accuracy ({folds}-fold cross validation)', fontsize=14)
plt.ylim([0.5,0.9])

plt.title(f'Mean accuracy versus min sample leaf ({folds}-fold cross-validation)', fontsize=18)

plt.show()


# ## 3.2.5. Post-pruning with cost complexity ##

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data_train_normal, labels_train, random_state=0)

#get alpha values (ccp_alphas)
clf = DecisionTreeClassifier(random_state=0)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

#build trees for each alpha values
clfs = []
for ccp_alpha in ccp_alphas:
  clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
  clf.fit(X_train, y_train)
  clfs.append(clf)

#get rid of the tree with one node only
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

#The effect of alpha on accuracy in train and test sets
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig = plt.figure(figsize=(12,8))
plt.xlabel("alpha", fontsize=14)
plt.ylabel("accuracy", fontsize=14)
plt.title("Accuracy versus alpha for training and testing sets", fontsize=18)
plt.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
plt.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
plt.legend()
plt.show()


# In[ ]:


ccp_alphas[ccp_alphas>0.003]


# In[ ]:


optimum_alpha = 0.00479408


# ## 3.2.6. Build the final (optimal) tree

# In[ ]:


#without feature selection

X_train, X_test, y_train, y_test = train_test_split(data_train_normal, labels_train, test_size=0.3, random_state=101)
#dtree = DecisionTreeClassifier(max_depth = optimum_depth, ccp_alpha = optimum_alpha)
dtree = DecisionTreeClassifier(ccp_alpha = optimum_alpha)

print(dtree.fit(X_train, y_train))

train_accuracy = dtree.score(X_train, y_train)

predictions = dtree.predict(X_test)
print('Training accuracy:', train_accuracy)

print('\nTesting')
print('Confusion matrix:\n', confusion_matrix(y_test,predictions))
print('\n')
report = classification_report(y_test,predictions)
print(classification_report(y_test,predictions))


# In[ ]:


#RFE with default n_features_to_select (half of total features) and cross-validation

X = data_train_normal
y = labels_train

# create pipeline
rfe = RFE(estimator=DecisionTreeClassifier(ccp_alpha = optimum_alpha))
model = DecisionTreeClassifier(ccp_alpha = optimum_alpha)
pipeline = Pipeline(steps=[('s',rfe),('m',model)])
# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# In[ ]:


#Final tree
rfe_tree = RFE(estimator=DecisionTreeClassifier(ccp_alpha = optimum_alpha))
rfe_tree.fit(X, y)
# summarize all features
for i in range(X.shape[1]):
	print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe_tree.support_[i], rfe_tree.ranking_[i]))

#let's store the selected features for later 
tree_selected_features = rfe_tree.support_


# Let's visualize which are the selected features for Desicion Tree classification
# 
# yellow -> selected

# In[ ]:


ax = plt.axes()
#sns.heatmap(rfe_tree.support_.reshape(1,-1),annot=True,yticklabels=False,cbar=False,cmap='viridis') #annotation looks really bad
sns.heatmap(rfe_tree.support_.reshape(1,-1),yticklabels=False,cbar=False,cmap='viridis')
ax.set_title('Features selected for Decision Tree')
plt.show()


# In[ ]:




