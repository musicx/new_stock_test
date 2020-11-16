import numpy as np
import pandas as pd
from pymodline.labeled import forest_trainer as ft
from pymodline import model_loader as ml
import pymodline.eval.score_evaluator as eva
import pymodline.eval.pivot_reporter as rpt

TRAIN = 1

if TRAIN:
    train = pd.read_hdf('../data/indi_train.hdf', 'data')
    print('training data read')
    mod, vals = ft.train_random_forest(train, 'up', ignores=['date', 'code', 'mid'], num_trees=1000, samples_split=10, sample_leaf=5, magic_number=206)
    print('model trained')

    for name, imp in zip(vals, mod.feature_importances_):
        print('{}: {}'.format(name, imp))

    ft.save_random_forest(mod, 'ind', base_folder='..', feature_names=vals)
    print('model saved')

else:
    #mod, val = ml.load_model('../models/random_forest_ind_T_200_V_3150_D_20_MS_10_ML_5_R_518.pkl',
    #                         '../features/random_forest_ind_T_200_V_3150_D_20_MS_10_ML_5_R_518_features.csv')
    pass

test = pd.read_hdf('../data/indi_test.hdf', 'data')
print('test data read')

fut = test.loc[test.date > '2018-05-10', :]
test = test.loc[test.date <= '2018-04-20', :]
print(test.shape)
print(fut.shape)
pred = ft.score(test, mod, vals, 'up', ['date', 'code', 'up', 'mid'])
print('score predicted')

data = eva.prepare_data(pred, 'up')
sl = eva.check_single_score(data, 'prob_of_1', 'up')
op = eva.check_operation_points(sl, points=1000)
ocm = eva.analyze_performance(op, 1)
# rpt.create_tabular_report('../data/rf_report.csv', ocm)
ocm.to_csv("../data/rf_report.csv")
print('done')

fut_pred = ft.score(fut, mod, vals, 'up', ['date', 'code'])
fut_pred.to_csv('../data/rf_predict.csv')
