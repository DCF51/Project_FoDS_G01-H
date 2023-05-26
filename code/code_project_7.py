# ______________________________  Code for the Project of Group G01-H ______________________________

# ==============================  Imports/Packages  ==============================
# standard packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# data split and standardizing(/scaling)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, KFold #je nachdem ob wir KFold brauchen, das weglassen
from sklearn.preprocessing import StandardScaler
# the 4 learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

    print(':) Successfully implemented the confusion matrix!')

    print('Confusion matrix:\n\t  |y_true = 0\t|y_true = 1')
    print('----------|-------------|------------')
    print('y_pred = 0|  ' + str(tp) + '\t\t|' + str(fp))
    print('y_pred = 1|  ' + str(fn) + '\t\t|' + str(tn))

    # Check for denominator of zero
    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator != 0 else 0

    # Calculate the evaluation metrics
    precision = safe_divide(tp, (tp + fp))
    recall = safe_divide(tp, (tp + fn))
    f1_score = safe_divide((tp), (tp + (0.5*(fn + fp))))
    specificity = safe_divide(tn, (tn + fp))
    accuracy = safe_divide((tp+tn), (tp + tn + fp + fn))

    # Get the roc curve using a sklearn function
    y_test_predict_proba  = clf.predict_proba(X)
    fp_rates, tp_rates, _ = roc_curve(y, y_test_predict_proba[:,1]) # i want the predictioin probability for the class "1"

    # Calculate the area under the roc curve using a sklearn function
    roc_auc = auc(fp_rates, tp_rates)

    # Plot on the provided axis - feel free to make this plot nicer if
    # you want to.
    ax.plot(fp_rates, tp_rates, label = 'Fold {}'.format(legend_entry))

    return [accuracy, precision, recall, specificity, f1_score, roc_auc]

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
#data = data.dropna(subset=['bmi']) #This drops the rows which have a missing value in bmi 
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

# ==============================  Data manipulation  ==============================
# Identify the categorical (cat_cols), numerical features (num_cols) and boolean features(boolean_cols)
num_cols = ['age', 'avg_glucose_level', 'bmi']
cate_cols = ['gender', 'work_type', 'Residence_type', 'smoking_status']
boolean_cols = ['hypertension', 'heart_disease', 'ever_married', 'stroke']

# Convert boolean columns to integers (0 or 1)
data[boolean_cols] = data[boolean_cols].astype(int)

# This replaces the categorical columns with their corresponding encoded numerical values
label_encoder = LabelEncoder()
for col in cate_cols:
    data[col] = label_encoder.fit_transform(data[col])
# is this really better than one-hot encoding ? since there is no order in our categorical variables it should not matter which one we use?

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
# has to be after data split on training data only

# ==============================  Data split  ==============================

X = data.copy().drop(columns='stroke')
y = data['stroke']

# ––––––––––––––––––––––––––––––  SStratified K-fold cross validator  ––––––––––––––––––––––––––––––
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
accuracy_scores = []

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
'''#create copies like in the tutorial to avoid inplace operations
X_train_sc, X_test_sc = X_train.copy(), X_test.copy()

X_train_sc[num_cols] = sc.fit_transform(X_train[num_cols])
X_test_sc[num_cols] = sc.transform(X_test[num_cols])
'''
# ==============================  Models  ==============================
# Be careful that we have different names for our different dataframes (e.g. X_train_LR and X_train_RF)

# ––––––––––––––––––––––––––––––  Logistic Regression  ––––––––––––––––––––––––––––––
LR = LogisticRegression()

# ––––––––––––––––––––––––––––––  Random Forest  ––––––––––––––––––––––––––––––
RF = RandomForestClassifier()

# ––––––––––––––––––––––––––––––  Decision Tree  ––––––––––––––––––––––––––––––
# Hyperparameters for Decision Tree Classifier (for a better overview)
DT = DecisionTreeClassifier(random_state=42)

# Hyperparameters for Grid Search Crossvalidator
# Create a prameter grid for the max_depth and criterion hyperparameters and save these in a dictionary
criterion   = ['gini', 'entropy']
max_depth   = np.arange(5, 21)  # min depth should not be too small (at 1 it will chose this for almost all trees, probably due to overfitting???)
parameters  = dict(criterion = criterion,
                  max_depth = max_depth)

# Prepare the performance overview data frame
df_performance_DT = pd.DataFrame(columns = ['fold','accuracy','precision','recall','specificity','F1','roc_auc'])

# Counter to keep track of fold number
fold = 0

# Creating figure
fig_DT,ax_DT = plt.subplots(1,1,figsize=(9, 9))

for train_i, test_i in kfold.split(X, y):
    print('Working on fold ', fold)

    # Get the relevant subsets for training and testing
    X_test_DT  = X.iloc[test_i]
    y_test_DT  = y.iloc[test_i]
    X_train_DT = X.iloc[train_i]
    y_train_DT = y.iloc[train_i]

    # Standardizing of numerical columns
    X_train_DT_sc = sc.fit_transform(X_train_DT)
    X_test_DT_sc = sc.transform(X_test_DT)

    # Train the model with HP tuning
    GS = GridSearchCV(DT, parameters, cv=10)
    GS.fit(X_train_DT_sc, y_train_DT)
    print('Best Criterion:', GS.best_estimator_.get_params()['criterion'])
    print('Best max_depth:', GS.best_estimator_.get_params()['max_depth'])

    # Fit the Desicion Tree Classifier with the best hyperparameters (didnt work without fitting again)
    DT_best = DecisionTreeClassifier(criterion=GS.best_estimator_.get_params()['criterion'],
                                     max_depth=GS.best_estimator_.get_params()['max_depth'])
    DT_best.fit(X_train_DT_sc, y_train_DT)

    # Evaluate the classifier
    eval_metrics_DT = evaluation_metrics(DT_best, y_test_DT, X_test_DT_sc, ax_DT, legend_entry=str(fold+1))
    df_performance_DT.loc[len(df_performance_DT), :] = [fold] + eval_metrics_DT

    # Plot the decision tree
    fig = plt.figure(figsize=(50,50)) #use dpi=... to increase resolution
    DT_plot = plot_tree(GS.best_estimator_,
                    feature_names=X.columns,
                    class_names=X.index.astype(str),
                    filled=True, 
                    label=str(fold+1))

    # increase counter for folds
    fold += 1

# Edit the ROC-AUC plot
ax_DT.plot([0, 1], [0, 1], color='r', ls='--', label='Random Classifier\n(AUC = 0.5)')
ax_DT.axis('square')
ax_DT.set_xlabel('FPR')
ax_DT.set_ylabel('TPR')
ax_DT.set_title('Decision Tree – Receiver Operating Characteristic')
ax_DT.legend()
ax_DT.grid(visible=True, which='major', axis='both', color='grey', linestyle=':', linewidth=1)
plt.tight_layout()

# ––––––––––––––––––––––––––––––  Support Vector Machine  ––––––––––––––––––––––––––––––
SVM = SVC() #or NuSVC()/ LinearSVC(), for Moreno to decide which one fits best