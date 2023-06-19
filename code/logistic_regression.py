#!/usr/bin/env python
# coding: utf-8

# # Code for the Project of Group G01-H 

# # Imports/Packages

# In[1]:


# standard packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# data split and standardizing(/scaling)
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
# the 4 learning models
from sklearn.linear_model import LogisticRegression
# metrics
from sklearn.metrics import r2_score, mean_squared_error, roc_curve, confusion_matrix, auc, precision_score, recall_score, f1_score
#Label encoder
from sklearn.preprocessing import LabelEncoder
# other
from collections import Counter
from sklearn.metrics import ConfusionMatrixDisplay
from imblearn.combine import SMOTEENN


import warnings

warnings.filterwarnings('ignore')


# # Functions

# In[2]:


# Utility function to plot the diagonal line
def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

def evaluation_metrics(clf, y, X,legend_entry='my legendEntry'):
    """
    compute multiple evaluation metrics for the provided classifier given the true labels
    and input features. Provides a plot of the roc curve on the given axis with the legend
    entry for this plot being specified, too.

    :param clf: classifier method
    :type clf: numpy array

    :param y: true class labels
    :type y: numpy array

    :param X: feature matrix
    :type X: numpy array

    :param ax: matplotlib axis to plot on
    :type legend_entry: matplotlib Axes

    :param legend_entry: the legend entry that should be displayed on the plot
    :type legend_entry: string

    :return: confusion matrix comprising the
             true positives (tp),
             true negatives  (tn),
             false positives (fp),
             and false negatives (fn)
    :rtype: four integers
    """

    # Get the label predictions
    y_test_pred    = clf.predict(X)

    # Calculate the confusion matrix given the predicted and true labels
    tn, fp, fn, tp = confusion_matrix(y, y_test_pred).ravel()
    
    print(':) Successfully implemented the confusion matrix!')

    print('Confusion matrix:\n\t  |y_true = 0\t|y_true = 1')
    print('----------|-------------|------------')
    print('y_pred = 0|  ' + str(tn) + '\t|' + str(fn))
    print('y_pred = 1|  ' + str(fp) + '\t\t|' + str(tp))
    
    confusion_matrix_values = np.array([[tn, fp], [fn, tp]])

    # Create a heatmap using seaborn
    ax = sns.heatmap(confusion_matrix_values, annot=True, fmt="d", cmap="Blues", cbar=True)

    # Set labels, title, and ticks
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(["Negative", "Positive"])
    ax.yaxis.set_ticklabels(["Negative", "Positive"])

    # Show the plot
    plt.show()

    # Check for denominator of zero
    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator != 0 else 0

    # Calculate the evaluation metrics
    precision = safe_divide(tp, (tp + fp))
    recall = safe_divide(tp, (tp + fn))
    f1_score = safe_divide((tp), (tp + (0.5*(fn + fp))))
    specificity = safe_divide(tn, (tn + fp))
    accuracy = safe_divide((tp+tn), (tp + tn + fp + fn))

    return [accuracy, precision, recall, specificity, f1_score]


# # Import data

# In[3]:


data = pd.read_csv(
    filepath_or_buffer='../data/healthcare-dataset-stroke-data.csv',
    index_col='id',
    dtype= {'gender':object, 'age':float, 'hypertension':bool, 'heart_disease':bool,
            'ever_married':object, 'work_type':object, 'Residence_type':object, 
            'avg_glucose_level':float, 'bmi':float, 'smoking_status':object, 'stroke':bool}
            )


# In[4]:


display(data)


# # Define datatypes

# In[5]:


#as categorical if needed
data['gender'] = pd.Categorical(data['gender'])
data['work_type'] = pd.Categorical(data['work_type'])
data['Residence_type'] = pd.Categorical(data['Residence_type'])
data['smoking_status'] = pd.Categorical(data['smoking_status'])

#change 'ever_married' to boolean
data['ever_married'] = data['ever_married'].map({'Yes': True, 'No': False})
data['ever_married']=data['ever_married'].astype(bool)


# # Data description

# In[6]:


# Shape and meaning of dataframe -- df.info(), df.shape[], df.columns, df.head()
data.info()
print('There are ', data.shape[1],'columns in the data.')
print('There are ', data.shape[0],'rows in the data.')

#data.columns not necessary because already included in data.info()

# Datatypes -- df.info() and df.dtypes
#data.dtypes not necessary because already included in data.info()

# Missing Data -- df.isna().sum()
print('missing values:')
print(data.isna().sum()) ###bmi has 201 missing values
# Calculate the mean or median of the "bmi" feature
bmi_mean = data['bmi'].mean()
bmi_median = data['bmi'].median()
print(bmi_mean)
print(bmi_median)
# Replace the missing values with the mean or median
data['bmi'].fillna(bmi_mean, inplace=True) 
print(data.isna().sum())

# Brief summary of extremes/means/medians -- df.describe()
print(data.describe())

# Check for duplicate rows -- df.duplicated()
print('sum of duplicated lines is:', data.duplicated().sum()) #marks all duplicates except the first occurence


# # Data manipulation

# In[7]:


# Identify the categorical (cat_cols), numerical features (num_cols) and boolean features (boolean_cols)
num_cols = ['age', 'avg_glucose_level', 'bmi']
cate_cols = ['gender', 'work_type', 'Residence_type', 'smoking_status']
boolean_cols = ['hypertension', 'heart_disease', 'ever_married', 'stroke']

# Convert boolean columns to integers (0 or 1)
data[boolean_cols] = data[boolean_cols].astype(int)

# This replaces the categorical columns with their corresponding encoded numerical values
label_encoder = LabelEncoder()
for col in cate_cols:
    data[col] = label_encoder.fit_transform(data[col])

#I dont know if the mapping is important but it could be because some machine learning algorithms can interprete data better with mapping but i am not to familiar with this point. 
for col in boolean_cols:
    unique_values = data[col].unique()
    #mapping = {value: index for index, value in enumerate(unique_values)}
    mapping = {value: int(value) for value in unique_values}
    data[col] = data[col].map(mapping)
    
# This helps that no more columns are created than before
columns_to_keep = num_cols + cate_cols + boolean_cols
data_e = data[columns_to_keep]


# In[8]:


display(data) #displays the data after the data manipulation


# # Correlation matrix

# In[9]:


# compute the correlation matrix
correlation = data.corr()
mask = np.triu(np.ones_like(correlation,dtype = bool))
fig, ax = plt.subplots(figsize = (11, 9)) #setting the size of the plot
cmap = sns.diverging_palette(230, 20, as_cmap=True) #selects color palette
sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=.8, center=0, square=True, linewidths=.8, cbar_kws={'shrink':.5})
plt.savefig("../output/correlation_matrix.pdf", backend="pdf", dpi = 300, bbox_inches='tight')


# # Feature selection

# In[10]:


df = data.drop('ever_married', axis = 1) #drops the column ever_married since it highly correlates with age as seen in the correlation matrix


# # Data Visualization

# In[11]:


sns.set_theme(style='darkgrid')
ax = sns.countplot(data=data, x = 'smoking_status')
xtick_labels = ['unknown', 'formerly smoked', 'never smoked', 'smokes']
# Set x-tick labels
ax.set_xticklabels(xtick_labels)
ax.set_title('Smoking Status')
plt.show()


# In[12]:


age_distribution = sns.histplot(x = data['age'], color = 'blue', bins=1, binwidth=1)
age_distribution.set(xlabel = 'Age', ylabel = 'Number of Patients', title = 'Age Distribution')
plt.grid(True)
age_distribution.set_axisbelow(True)
age_distribution.set_zorder(-1)


# In[13]:


bmi_distribution = sns.histplot(x = data['bmi'], color = 'blue', bins=1, binwidth=1)
bmi_distribution.set(xlabel = 'BMI', ylabel = 'Number of Patients', title = 'BMI Distribution')
plt.grid(True)
bmi_distribution.set_axisbelow(True)
bmi_distribution.set_zorder(-1)


# In[14]:


glucose_level_distribution = sns.histplot(x = data['avg_glucose_level'], color = 'blue', bins=1, binwidth=1)
glucose_level_distribution.set(xlabel = 'Average Glucose Level', ylabel = 'Number of Patients', title = 'Glucose Level Distribution')
plt.grid(True)
glucose_level_distribution.set_axisbelow(True)
glucose_level_distribution.set_zorder(-1)


# # Data split

# In[15]:


X = df.copy().drop(columns='stroke')
y = df['stroke']


# # Normal Split or K-Fold?

# In[16]:


class_counts = df['stroke'].value_counts()
class_proportions = df['stroke'].value_counts(normalize=True)

print("Class Counts:")
print(class_counts)

print("\nClass Proportions:")
print(class_proportions)


# So we can see there are 4861 of non-stroke and 249 with stroke. So the majority has no stroke. In the class proportions we can see the porportion of 'Flase' is 0.9513 --> 95.13% of the data. The porportion of 'True' is 0.0487 --> 4.87% of the data
# ==>highly imbalanced: So stratified K-Fold Cross-Valdidator would be the better option

# # Balance the data

# In[17]:


#With the help of Smote we perform a resampling
# Perform resampling
sampler = SMOTEENN(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X, y)

# Check the class distribution after resampling
print(pd.Series(y_resampled).value_counts())


# # Stratified K-fold cross validator

# In[18]:


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# # Model Implementation

# In[19]:


#Prepare the performance overview data frame
df_performance = pd.DataFrame(columns = ['fold','clf','accuracy','precision','recall',
                                         'specificity','F1'])

all_importance = pd.DataFrame(index=range(0, 5), columns=X.columns)

# Use this counter to save your performance metrics for each crossvalidation fold
# also plot the roc curve for each model and fold into a joint subplot
fold = 0

# Be careful that we have different names for our different dataframes (e.g. X_train_LR and X_train_RF)

for train_index, test_index in kfold.split(X, y):
    X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
    y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
    
    # ----------------------------  Standardizing/Scaling  ---------------------------------
    sc = StandardScaler()
    #create copies like in the tutorial to avoid inplace operations
    X_train_sc, X_test_sc = X_train.copy(), X_test.copy()

    X_train_sc[num_cols] = sc.fit_transform(X_train[num_cols])
    X_test_sc[num_cols] = sc.transform(X_test[num_cols])
   
    # ––––––––––––––––––––––––––––––  Logistic Regression  ––––––––––––––––––––––––––––––
    log_reg_classifier = LogisticRegression()
    log_reg_classifier.fit(X_train_sc, y_train)
    
    # Calculate feature importance
    coefficients = log_reg_classifier.coef_
    importance = np.abs(coefficients)
    importance_scores = (importance / np.sum(importance))
    all_importance.loc[fold] = importance_scores
    
    
    # ––––––––––––––––––––––––––––––  Evaluate your classifiers  ––––––––––––––––––––––––––––––
    eval_metrics = evaluation_metrics(log_reg_classifier, y_test, X_test_sc,legend_entry=str(fold))
    df_performance.loc[len(df_performance),:] = [fold,'Logistic Regression'] + eval_metrics
    
    # increase counter for folds
    fold += 1


# # Receiver Operating Characteristic (ROC) curve

# In[20]:


plt.figure()
fold = 0
for train_index, test_index in kfold.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # ----------------------------  Standardizing/Scaling  ---------------------------------
    sc = StandardScaler()
    #create copies like in the tutorial to avoid inplace operations
    X_train_sc, X_test_sc = X_train.copy(), X_test.copy()

    X_train_sc[num_cols] = sc.fit_transform(X_train[num_cols])
    X_test_sc[num_cols] = sc.transform(X_test[num_cols])

    y_scores = log_reg_classifier.predict_proba(X_test_sc)[:, 1]

    # Calculate the false positive rate (fpr), true positive rate (tpr), and thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    # Calculate the area under the ROC curve (AUC)
    roc_auc = auc(fpr, tpr)
    
    # Plot the ROC curve for the current fold
    plt.plot(fpr, tpr, label='Fold {} (AUC = {:.2f})'.format(fold+1, roc_auc))
    
    # increase counter for folds
    fold += 1

# Plot the random classifier line
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True, linestyle = "--", linewidth = 0.5, alpha = 0.7)


#save plot
plt.savefig("../output/roc_curve_log_reg.pdf", backend="pdf", dpi = 300, bbox_inches='tight')

# Show the plot
plt.show()


# # Evaluation Metrics

# In[21]:


df_performance
#recall = sensitivity = true positive rate
#precision = proportion of true positive predicted values out of all positive predictions


# # Summarize the folds

# In[22]:


print(df_performance.groupby(by = 'clf').agg(['mean', 'std']))


# # Feature Importance - Logistic Regression

# In[23]:


display(all_importance)


# In[24]:


# Visualize the normalized feature importance across the five folds and add error bar to indicate the std
fig, ax = plt.subplots(figsize=(18, 6))

ax.bar(np.arange(importance_scores.shape[1]), all_importance.mean(), yerr=all_importance.std(), capsize=3)
ax.set_xticks(np.arange(importance_scores.shape[1]), X.columns.tolist(), rotation=45, fontsize=16)
ax.set_title("Normalized feature importance for LR across 5 folds", fontsize=20)
plt.xlabel('Risk Factors', fontsize=16)
plt.ylabel("Normalized feature importance", fontsize=16)
plt.savefig("../output/feature_importance_log_reg.pdf", backend="pdf", dpi = 300, bbox_inches='tight')
plt.show()

