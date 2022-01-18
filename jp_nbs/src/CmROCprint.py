from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

def cm_seaborn_per(input_cm):
    """
    input_cm= a numpy array with (2,2) shape; 
    TN= input_cm[0,0]
    TP= input_cm[1,1]
    """
    negative_values= np.round(input_cm.flatten()[0:2]/np.sum(input_cm.flatten()[0:2])*100, 2) 
    positive_values= np.round(input_cm.flatten()[2:5]/np.sum(input_cm.flatten()[2:5])*100, 2) 
    final_per_values= np.array([negative_values, positive_values]).reshape(2,2)
    return final_per_values

def cm_seaborn_labels(input_cm):
    """
    input_cm= a numpy array with (2,2) shape; 
    TN= input_cm[0,0]
    TP= input_cm[1,1]
    """
    
    cm_class= ['TN', 'FP', 'FN', 'TP']
    cm_counts= ["{0:0.0f}".format(i) for i in input_cm.flatten()]
    negative_values= input_cm.flatten()[0:2]/np.sum(input_cm.flatten()[0:2])
    positive_values= input_cm.flatten()[2:5]/np.sum(input_cm.flatten()[2:5])
    percentage= ["{0:.2%}".format(i) for i in np.append(negative_values, positive_values)]
    labels= [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(cm_counts, percentage, cm_class)]
    labels= np.asarray(labels).reshape(2,2)
    return labels

def plot_roc_cm(model, cv, X, y, verbose=False, roc_title="ROC curve", cm_title="Confusion matrix",save=False, save_path=''):
    """
    Draw a Cross Validated ROC Curve and Confusion matrix.
    Args: 
        model= XGBoost, logistic regression;
        cv= crossvalidation method
        verbose= if you want to print proportions and folds
    This function has been complemented from here: https://stackoverflow.com/questions/29656550/how-to-plot-pr-curve-over-10-folds-of-cross-validation-in-scikit-learn
    """
    fig, (ax1, ax2)= plt.subplots(nrows=1, ncols=2, figsize=(20,10), gridspec_kw= {'width_ratios': [1.5, 1]})
    
    i=1
    # ROC curve variables
    mean_fpr= np.linspace(0,1,100)
    values_auc= []; values_tpr= []
    color_values= ['red', 'blue', 'green']

    # CM variables
    final_tp= []; final_fp= []
    final_tn = []; final_fn= []
    
    for train, test in cv.split(X, y.values.ravel()):
        fit_model= model.fit(X.iloc[train], y.iloc[train].values.ravel())
        # ROC curve:
        prediction= fit_model.predict_proba(X.iloc[test])[:,1] #raw probabilities
        [fpr, tpr, t]= roc_curve(y.iloc[test].values.ravel(), prediction)
        values_tpr.append(np.interp(mean_fpr, fpr, tpr))
        roc_auc= auc(fpr, tpr)
        values_auc.append(roc_auc)
        # CM
        y_pred= fit_model.predict(X.iloc[test])
        cm= confusion_matrix(y.iloc[test], y_pred)
        final_tp.append(cm[0][0]); final_fp.append(cm[0][1])
        final_tn.append(cm[1][1]); final_fn.append(cm[1][0])
        if i <= 10:
            color= color_values[0]
        elif i > 10 and i <= 20:
            color= color_values[1]
        else:
            color= color_values[2]
        #Plot each ROC K-fold
        ax1.plot(fpr, tpr, lw=2, alpha= 0.2, color= color, label= 'AUROC= %0.2f, fold %d' % (roc_auc, i))
        if verbose:
            train_prop_values = Counter(y.iloc[train].values.ravel())
            test_prop_values= Counter(y.iloc[test].values.ravel())
            print(i)
            print("Train: ", X.iloc[train].shape, "Test: ", y.iloc[test].shape)
            print("Train hit proportion: ", train_prop_values, np.round(train_prop_values[1]/train_prop_values[0],2))
            print("Test hit proportion: ", test_prop_values, np.round(test_prop_values[1]/test_prop_values[0],2))
            print("-"*10)
        i += 1
    # Plot mean ROC values
    ax1.plot([0,1],[0,1],linestyle = '--', lw = 2, color = 'black', alpha=0.7)
    mean_tpr= np.mean(values_tpr, axis=0)
    mean_auc= auc(mean_fpr, mean_tpr)
    ax1.plot(mean_fpr, mean_tpr, color='black', lw= 2, label=r'Mean AUROC= %0.4f' % (mean_auc))

    # Plot std of mean AUROC
    std_tpr= np.std(values_tpr, axis=0)
    tprs_upper= np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower= np.maximum(mean_tpr - std_tpr, 0)
    ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.4, label=r'$\pm$ 1 std. dev.')

    # Plot aesthetics
    ax1.set(xlabel="False positive rate (1 - Specificity)", ylabel= "True positive rate (Sensitivity)")
    ax1.set_title(roc_title)
    ax1.legend(loc="lower right")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Plot CM
    final_cm= np.array([np.mean(final_tp), np.mean(final_fp), 
                    np.mean(final_fn), np.mean(final_tn)]).reshape(2,2)
    
    cm_per= cm_seaborn_per(input_cm= final_cm)
    labels= cm_seaborn_labels(input_cm= final_cm)
    tick_labels= ['Not hit', 'Hit']
    sns.heatmap(cm_per, annot= labels, fmt= "", cmap= "Blues", xticklabels= tick_labels, yticklabels= tick_labels)
    ax2.set_title(cm_title)
    ax2.set(xlabel="Predicted label", ylabel= "True label")
    
    plt.show()
    fig.tight_layout()
    if save:
        fig.savefig(save_path, transparent= True, bbox_inches= 'tight')

def plot_roc_cm_final(model, X, y, roc_title="ROC curve", cm_title= "Confusion matrix", save= False, save_path= ""):
    """
    This function is to test it in all the dataset
    model= XGBoost, etc.
    """
    fig, (ax1, ax2)= plt.subplots(nrows=1, ncols=2, figsize=(20,10), gridspec_kw= {'width_ratios': [1.5, 1]})
    
    # ROC curve:
    prediction= model.predict_proba(X)[:,1]
    [fpr, tpr, t]= roc_curve(y.values.ravel(), prediction)
    roc_auc= auc(fpr, tpr)
    ax1.plot(fpr, tpr, lw=2, alpha= 1, label= 'ROC (AUC = %0.4f)' % (roc_auc))
    ax1.plot([0,1],[0,1],linestyle = '--', lw = 2, color = 'black', alpha=0.7)
    ax1.set(xlabel="False positive rate (1 - Specificity)", ylabel= "True positive rate (Sensitivity)")
    ax1.set_title(roc_title)
    ax1.legend(loc="lower right")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # CM
    cm_predict= model.predict(X)
    cm= confusion_matrix(y, cm_predict)
    cm_per= cm_seaborn_per(input_cm= cm)
    labels= cm_seaborn_labels(cm)
    tick_labels= ['Not hit', 'Hit']
    sns.heatmap(cm_per, annot= labels, fmt= "", cmap= "Blues", xticklabels= tick_labels, yticklabels= tick_labels)
    ax2.set(xlabel="Predicted label", ylabel= "True label")
    ax2.set_title(cm_title)
    
    plt.show()
    fig.tight_layout()
    if save:
        fig.savefig(save_path, transparent= True, bbox_inches= 'tight')


