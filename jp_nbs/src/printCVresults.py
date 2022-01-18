#!/usr/bin/env python
# coding: utf-8
import numpy as np

def print_results_cross_validate(scoring_dict, input_model, cross_validate_results, print_last=True):
    """
    scoring_dict: dictionary with keys print and values evaluation metrics
    input_model: model
    cross_validate_results: dictornary from sklearn.model_selection.cross_validate
    """
    print("-"*10)
    for i in range(len(scoring_dict)):
        print(input_model.__class__.__name__ + " mean %s: %.4f (+/- %.4f)" 
              %(list(scoring_dict.keys())[i], 
                cross_validate_results['test_%s' % list(scoring_dict.values())[i]].mean(),
                cross_validate_results['test_%s' % list(scoring_dict.values())[i]].std()
               )
             )
    if print_last:
        print("-"*10)

