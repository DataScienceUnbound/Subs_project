__author__ = 'Jonathan'

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as sk_metrics

def check_df_col_feature_types(X):
    """
    Loop through a dataframe and check column types
    """
    for i,series in X.iteritems():
        print(series.name,':', type_of_target(series),'of type', series.dtype)
        print(series.value_counts()[:5])
        print('****')

def get_feature_imps(clf_obj, X_df):
    """ Utility function to return feature importances from models"""
    # todo - check that clf_object has the feature_importances_ attributes
    feature_imps = pd.Series(data=clf_obj.feature_importances_, index=X_df.columns).sort_values(ascending=False)
    return feature_imps

def get_preds_from_rf_oob(rf):
    # todo docstring

    dummy_preds = np.zeros(shape=rf.oob_decision_function_.shape)
    dummy_preds[np.arange(rf.oob_decision_function_.shape[0]), rf.oob_decision_function_.argmax(axis = 1)] = 1
    dummy_preds_df = pd.DataFrame(dummy_preds, columns = rf.classes_).astype(int)

    yhat = dummy_preds_df.idxmax(axis=1)
    return yhat

def labelled_confusion_matrix(y, ypred, clf):
    # todo docstring
    row_labels = ['True '     + label for label in clf.classes_]
    col_labels = ['Predicted '+ label for label in clf.classes_]

    cm = confusion_matrix(y,ypred,labels= clf.classes_)
    cm_df = pd.DataFrame(cm, index=row_labels, columns=col_labels)

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.round(cm_norm, decimals=3)
    cm_norm_df = pd.DataFrame(cm_norm, index=row_labels, columns=col_labels)

    return cm_df, cm_norm_df

def get_model_performance(y, x, model, n_cv = 3, test_split=0.1,random_state = 7):
    """
    # todo docstring

    Arguments:
    - y
    - x: should we assert to have to be dataframe?
    - test_split: float or None. If None no data is used for split
       and no test split is reported, just CV.
    """

    metrics = [r2_score, mean_absolute_error, mean_squared_error]

    # first split off test and training set, x is now non test, not full set
    if test_split:
        x, x_test, y, y_test = train_test_split(x,y,
                                                test_size=test_split,
                                                random_state=random_state)

    # the cross-validation on the training
    model_cv_performance = {m.__name__:[] for m in metrics}
    kfold_cv = KFold(n_cv, shuffle=True, random_state=random_state)
    for t_index, v_index in kfold_cv.split(x):
        x_val   = x.iloc[v_index,:].copy() # copy incase we get a view and then do pre-processing...
        x_train = x.iloc[t_index,:].copy()
        y_val   = y.iloc[v_index].copy()
        y_train = y.iloc[t_index].copy()

        # insert pre-processing function here

        model = model.fit(x_train, y_train)
        val_predictions = model.predict(x_val)
        # you could wrap up val predictions with a series to remember the book index etc

        for score_function in metrics:
            score = score_function(y_val, val_predictions)
            metric_str = score_function.__name__
            model_cv_performance[metric_str].append(score)

    for key in model_cv_performance.keys():
        model_cv_performance[key] = np.mean(model_cv_performance[key])
        # you want to return cross validation predictions too - might entail averaging if odd no.

    # now use same model on the test split, if test split. But train on full model
    # in future any parameters that are being varied should be chosen by cv
    if test_split:
        model = model.fit(x,y)
        test_predictions = model.predict(x_test)
        model_test_performances = {m.__name__:m(y_test, test_predictions) for m in metrics}

    print('Model Test performance:',    model_test_performances['r2_score'])
    print('Model CV mean performance:', model_cv_performance['r2_score'])


    return model

if __name__ == "__main__":
    pass