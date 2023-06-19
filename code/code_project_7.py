# ______________________________  Code for the Project of Group G01-H ______________________________

# ==============================  Imports/Packages  ==============================
# standard packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# data split and standardizing(/scaling)
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold #je nachdem ob wir KFold brauchen, das weglassen
from sklearn.preprocessing import StandardScaler
# the 4 learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
    # for decision trees see: https://scikit-learn.org/stable/modules/tree.html
from sklearn.svm import SVC, NuSVC, LinearSVC
    # for svm we have different methods see: https://scikit-learn.org/stable/modules/svm.html
# metrics
from sklearn.metrics import r2_score, mean_squared_error, roc_curve, confusion_matrix, auc
#Label encoder
from sklearn.preprocessing import LabelEncoder
# other
import warnings

warnings.filterwarnings('ignore')

# ==============================  Functions  ==============================
#only if needed (probably from homeworks/tutorials)

def evaluation_metrics(clf, y, X, ax,legend_entry='my legendEntry'):
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

    :return: comfusion matrix comprising the
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


    # Calculate the evaluation metrics
    precision   = tp/(tp+fp)
    specificity = tn/(tn+fp)
    accuracy    = (tp+tn)/(tp+tn+fp+fn)
    recall      = tp/(tp+fn)
    f1          = tp/(tp+0.5*(fn+fp))

    # Get the roc curve using a sklearn function
    y_test_predict_proba  = clf.predict_proba(X)
    fp_rates, tp_rates, _ = roc_curve(y, y_test_predict_proba[:,1]) # i want the predictioin probability for the class "1"

    # Calculate the area under the roc curve using a sklearn function
    roc_auc = auc(fp_rates, tp_rates)

    # Plot on the provided axis - feel free to make this plot nicer if
    # you want to.
    ax.plot(fp_rates, tp_rates, label = 'Fold {}'.format(legend_entry))

    return [accuracy,precision,recall,specificity,f1, roc_auc]


# ==============================  Import data  ==============================
data = pd.read_csv(
    filepath_or_buffer='../data/healthcare-dataset-stroke-data.csv',
    index_col='id',
    dtype= {'gender':object, 'age':float, 'hypertension':bool, 'heart_disease':bool,
            'ever_married':object, 'work_type':object, 'Residence_type':object, 
            'avg_glucose_level':float, 'bmi':float, 'smoking_status':object, 'stroke':bool}
            )

# ==============================  Define datatypes  ==============================
#as categorical if needed
data['gender'] = pd.Categorical(data['gender'])
data['work_type'] = pd.Categorical(data['work_type'])
data['Residence_type'] = pd.Categorical(data['Residence_type'])
data['smoking_status'] = pd.Categorical(data['smoking_status'])

#change 'ever_married' to boolean
data['ever_married'] = data['ever_married'].map({'Yes': True, 'No': False})
data['ever_married'].astype(bool)

# ==============================  Data description  ==============================
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
data = data.dropna(subset=['bmi']) #This drops the rows which have a missing value in bmi 

# Brief summary of extremes/means/medians -- df.describe()

print(data.describe())

# Check for duplicate rows -- df.duplicated()
print('sum of duplicated lines is:', data.duplicated().sum()) #marks all duplicates except the first occurence

# ==============================  Data manipulation  ==============================
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
    mapping = {value: index for index, value in enumerate(unique_values)}
    data[col] = data[col].map(mapping)
    
# This helps that no more columns are created than before
columns_to_keep = num_cols + cate_cols + boolean_cols
data_e = data[columns_to_keep]

# ==============================  Feature selection  ==============================
# don't know what exactly to use here yet

# ==============================  Data split  ==============================

X = data.copy().drop(columns='stroke')
y = data['stroke']

# We have two options for the split: "normal" train_test_split() or StratifiedKFold()
# ––––––––––––––––––––––––––––––  "Normal" split  ––––––––––––––––––––––––––––––
#Hyperparameters for "normal split"
test_size = 0.2

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=test_size)
print('Training set size: {}, test set size: {}'.format(len(X_train), len(X_test)))
# here no loop is required and Scaling can be done outside the loop

# ––––––––––––––––––––––––––––––  SStratified K-fold cross validator  ––––––––––––––––––––––––––––––
# Hyperparameters for the split and preparation (Stratified K-fold cross validator)
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

#loop over splits should then be done in the === Model === section using:
"""
for train_i, test_i in skf.split(X, y):

    # Get the relevant subsets for training and testing
    X_test  = X.iloc[test_i]
    y_test  = y.iloc[test_i]
    X_train = X.iloc[train_i]
    y_train = y.iloc[train_i]

    note: scaling has to be done inside the loop
"""
# ==============================  Standardizing/Scaling  ==============================
sc = StandardScaler()
#create copies like in the tutorial to avoid inplace operations
X_train_sc, X_test_sc = X_train.copy(), X_test.copy()

X_train_sc[num_cols] = sc.fit_transform(X_train[num_cols])
X_test_sc[num_cols] = sc.transform(X_test[num_cols])

# ==============================  Models  ==============================
# Be careful that we have different names for our different dataframes (e.g. X_train_LR and X_train_RF)

# ––––––––––––––––––––––––––––––  Logistic Regression  ––––––––––––––––––––––––––––––
LR = LogisticRegression()

# ––––––––––––––––––––––––––––––  Random Forest  ––––––––––––––––––––––––––––––
RF = RandomForestClassifier()

# ––––––––––––––––––––––––––––––  Decision Tree  ––––––––––––––––––––––––––––––
DT = DecisionTreeClassifier()

# ––––––––––––––––––––––––––––––  Support Vector Machine  ––––––––––––––––––––––––––––––
SVM = SVC() #or NuSVC()/ LinearSVC(), for Moreno to decide which one fits best
