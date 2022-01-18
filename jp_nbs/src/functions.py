import matplotlib.pyplot as plt
import numpy as np


def print_gs_results(gs_df):
    """
    Print GridSearchCV results
    """
    print('='*20)
    print("best params: " + str(gs_df.best_estimator_))
    print("best params: " + str(gs_df.best_params_))
    print('best score:', round(gs_df.best_score_, 4))
    print('='*20)

def auc_train_test_plot(scoring ,gs_df, x_size, y_size, x_min, x_max, grid_results):
    print_gs_results(gs_df= gs_df)
    plt.figure(figsize=(x_size, y_size))
    plt.title('GridSearch using 10-fold CV')
    plt.grid()
    plt.ylabel("AUC value")
    plt.xlabel("C: Inverse of regularization strength")
    ax= plt.axes()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0,1)
    X_axis=np.array(grid_results['param_C'].data, dtype= float)

    for scorer, color in zip(list(scoring.keys()), ['g']): 
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = -grid_results['mean_%s_%s' % (sample, scorer)] if scoring[scorer]=='neg_log_loss' else grid_results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = grid_results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std, 
                sample_score_mean + sample_score_std, 
                alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(grid_results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = -grid_results['mean_test_%s' % scorer][best_index] if scoring[scorer]=='neg_log_loss' else grid_results['mean_test_%s' % scorer][best_index]
        
        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))
    plt.legend(loc="best")
    plt.grid('off')
    plt.show()







