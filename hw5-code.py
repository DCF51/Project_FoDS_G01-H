# First import some packages we will be using - here we only import
# a subset of the packages you will need
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, auc
import matplotlib.pyplot as plt



### add additional imports from sklearn that you may need - no other package is needed!
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# Utility function to plot the diagonal line - complete, no input needed here.
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

# Functions you are asked to complete
def get_confusion_matrix(y,y_pred):
    """
    compute the confusion matrix of a classifier yielding
    predictions y_pred for the true class labels y
    :param y: true class labels
    :type y: numpy array

    :param y_pred: predicted class labels
    :type y_pred: numpy array

    :return: comfusion matrix comprising the
             true positives (tp),
             true negatives  (tn),
             false positives (fp),
             and false negatives (fn)
    :rtype: four integers
    """

    # true/false pos/neg. - this is a block of code that's needed
    # HINT: consider using a for loop.
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(y)):
        if y[i] == 1 and y_pred[i] == 1:
            tp += 1
        if y[i] == 0 and y_pred[i] == 1:
            fp += 1
        if y[i] == 0 and y_pred[i] == 0:
            tn += 1
        if y[i] == 1 and y_pred[i] == 0:
            fn += 1

    return tn, fp, fn, tp
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

    # Calculate the confusion matrix given the predicted and true labels with your function
    # only add the correct inputs here
    tn, fp, fn, tp = get_confusion_matrix(y, y_test_pred)

    # Ensure that you get correct values - this code will divert to
    # sklearn if your implementation fails - you can ignore the lines under
    # this comment, no input needed.
    tn_sk, fp_sk, fn_sk, tp_sk = confusion_matrix(y, y_test_pred).ravel()
    if np.sum([np.abs(tp-tp_sk) + np.abs(tn-tn_sk) + np.abs(fp-fp_sk) + np.abs(fn-fn_sk)]) >0:
        print('OWN confusion matrix failed!!! Reverting to sklearn.')
        tn = tn_sk
        tp = tp_sk
        fn = fn_sk
        fp = fp_sk
    else:
        print(':) Successfully implemented the confusion matrix!')

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


# Import data - note that upon submission please keep the original path
# '../data/data_subset_top100.csv' !! - no change needed
df = pd.read_csv('../data/data_subset_top100.csv',index_col=0)
X  = df.copy().drop('resistant', axis = 1)
y  = df['resistant']


# Get an overview over your dataframe
### a short block of code - plots are optional and not strictly required
print(df.shape)
print(df.dtypes)
print('Head function\n', df.head())

# Perform a 5-fold stratified crossvalidation - prepare the splitting
n_splits = 5
skf      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) #not sure if shuffle=False/True


# Prepare the performance overview data frame - keep this
df_performance = pd.DataFrame(columns = ['fold','clf','accuracy','precision','recall',
                                         'specificity','F1','roc_auc'])
df_LR_normcoef = pd.DataFrame(index = X.columns, columns = np.arange(n_splits))


# Use this counter to save your performance metrics for each crossvalidation fold
# also plot the roc curve for each model and fold into a joint subplot
fold = 0
fig,axs = plt.subplots(1,2,figsize=(9, 4))


# Loop over all splits
for train_i, test_i in skf.split(X, y):

    # Get the relevant subsets for training and testing
    X_test  = X.iloc[test_i]
    y_test  = y.iloc[test_i]
    X_train = X.iloc[train_i]
    y_train = y.iloc[train_i]


    # Standardize the numerical features using training set statistics
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc  = sc.transform(X_test)


    # Creat prediction models and fit them to the training data

    # Logistic regression
    clf = LogisticRegression()
    ### Fit
    clf.fit(X_train_sc, y_train)

    # Get the importance values - what part of the model do you need here?
    # We provided some skeleton below which should make saving these easier
    df_this_LR_coefs = pd.DataFrame(zip(X_train.columns, np.transpose(np.abs(clf.coef_))), columns=['features', 'coef'])
    df_LR_normcoef.loc[:,fold] = df_this_LR_coefs['coef'].array #converst a series to an array, so you are able to place the values

    # Random forest
    clf2 = RandomForestClassifier()
    ### Fit
    clf2.fit(X_train_sc, y_train)


    # Evaluate your classifiers - ensure to use the correct inputs
    eval_metrics = evaluation_metrics(clf, y_test, X_test_sc, axs[0], legend_entry=str(fold+1))
    df_performance.loc[len(df_performance), :] = [fold,'LR'] + eval_metrics

    eval_metrics_RF = evaluation_metrics(clf2, y_test, X_test_sc, axs[1], legend_entry=str(fold+1))
    df_performance.loc[len(df_performance), :] = [fold, 'RF'] + eval_metrics_RF

    # increase counter for folds
    fold += 1


# Edit the plot and make it nice
model_names = ['LR','RF']
for i,ax in enumerate(axs):
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    add_identity(ax, color="r", ls="--",label = 'random\nclassifier')
    ### - add other attributes to your plot - what is missing still?
    ax.legend()
    ax.set_title(model_names[i])
# Save the plot - no change needed - ensure to submit with this exact output path
plt.tight_layout()
plt.savefig('../output/roc_curves.png')


# Summarize the performance metrics over all folds
### this may be more than one line of code
df_performance_c = df_performance.copy().drop(columns='fold')
# the 2 df below: temporary dataframes for calculations
df_performance_cLR = df_performance_c.loc[df_performance['clf'] == 'LR'] 
df_performance_cRF = df_performance_c.loc[df_performance['clf'] == 'RF']
df_performance_s = pd.DataFrame(columns = ['clf','accuracy','precision','recall',
                                         'specificity','F1','roc_auc'], 
                                         index=[0, 1, 2, 3])
df_performance_s.loc[0, :] = df_performance_cLR.mean()
df_performance_s.loc[1, :] = df_performance_cLR.std()
df_performance_s.loc[2, :] = df_performance_cRF.mean()
df_performance_s.loc[3, :] = df_performance_cRF.std()
df_performance_s.loc[:, 'clf'] = ['LR', 'LR', 'RF', 'RF'] #set the classifier labels again because they were added too --> (LRLRLRLRLR, RFRFRFRFRF)
df_performance_s.loc[:, 'eval'] = ['mean', 'std', 'mean', 'std'] #describe what evaluation method is used in the rows

# export metrics as an excel table for the report
df_performance.to_excel('../output/metrics.xlsx')
df_performance_s.to_excel('../output/metrics_meanstd.xlsx')

# Average the feature importance across the five folds and sort them
# HINT: consider how to sort a pandas data frame
### this may be a short block of code
df_LR_normcoef_avg = pd.DataFrame(index = X.columns, columns=['avg', 'std'])
for i in X.columns:
    df_LR_normcoef_avg.loc[i, 'avg'] = df_LR_normcoef.loc[i, [0, 1, 2, 3, 4]].mean()
    df_LR_normcoef_avg.loc[i, 'std'] = df_LR_normcoef.loc[i, [0, 1, 2, 3, 4]].std()

df_LR_normcoef_avg_sorted = df_LR_normcoef_avg.sort_values(by='avg', ascending=False)

# Visualize the normalized feature importance averaged across the five folds
# FOR THE TOP 15 features and add error bars to indicate the std
#prepare indexer for top 15 features
top15 = pd.Series(data=df_LR_normcoef_avg_sorted.index).drop(index=np.arange(start=15, stop=100))

#create new dataframe (was the intention to use it in an attempt to get rid of an error), now used for simpler indexing in the next step
df_top15 = df_LR_normcoef_avg_sorted.loc[top15, :]

#use .astype to convert the pd.series to an actual size-1 array (Error before was: TypeError: only size-1 arrays can be converted to Python scalars)
top15avg = df_top15['avg'].astype(float)
top15std = df_top15['std'].astype(float)

fig, ax = plt.subplots(figsize=(18, 6))
ax.bar(np.arange(15), height=top15avg, yerr=top15std)
### add a short block of code to create a nice plot with all required labels etc.
ax.set_title('Normalized feature importance across 5 folds for the top 15 features')
ax.set_xticks(np.arange(15), top15, rotation=90)
fig.tight_layout()
fig.savefig('../output/importance.png')


# Get the two most important features
top2 = top15.drop(index=np.arange(2, 15))
top2f = pd.Series(data=df_top15.loc[top2, :].index)
print('The 2 most important features are: ', top2f.values)