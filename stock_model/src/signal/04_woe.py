# -*- coding: utf-8 -*-
import logging
import os
import pandas as pd
from pymodline.woe.bin_merger import BinMerger
from pymodline.woe.bin_splitter import BinSplitter


logging.basicConfig(format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)

file_names = os.listdir('../data/inds/')
train = []
test = []
for name in file_names[:2]:
    train_sub = pd.read_hdf('../data/inds/' + name, 'train')
    test_sub = pd.read_hdf('../data/inds/' + name, 'test')
    train.append(train_sub)
    test.append(test_sub)
train_data = pd.concat(train, axis=0).fillna(0).reset_index(drop=True)
test_data = pd.concat(test, axis=0).fillna(0).reset_index(drop=True)
train_data.drop(columns=['date'], inplace=True)
test_data.drop(columns=['date'], inplace=True)

splitter = BinSplitter(bad_name='go_up', # variables_to_check=['mtm_3_16_delta', 'mtmma_3_12_delta'],
                       min_observation=3, min_target_observation=0,
                       min_proportion=0, min_target_proportion=0, num_jobs=2)
print(splitter.precheck_data(train_data))

splitter.fit(train_data)
splitter.apply(test_data)
mrgr = BinMerger(min_drop=0, min_merge=0, min_merge_bad=0,
                 min_iv=0, max_miss=0.999999,
                 check_monotonicity=False, z_scale=False, num_jobs=2)
mrgr.fit(splitter.data_summary)

mrgr.save_variable_analysis('../data/inds/var_analysis.txt')
mrgr.save_woe_bins('../data/inds/woe_bins.txt')
