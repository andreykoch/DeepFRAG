#!/usr/bin/env python
# coding: utf-8
#
'''
Andrey Koch, Laboratory Corporation of America Holdings, All rights reserved.

This code is the core of the cancer prediction method DeepFRAG. It reproduces main results from the research article:
A. Koch, E. Giladi DeepFRAG: A method for cancer detection based on DNA fragmentomics and deep learning. Bioinformatic Advances, 2026, 10.1093/bioadv/vbag024.

Github:     https://github.com/andreykoch/DeepFRAG

'''
#
#
## import packages
#
import numpy as np
import pandas as pd
import random
import glob
import time
import pickle
import pywt
#
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
#
import tensorflow as tf
from tensorflow.keras import layers
#
#
## input data
#
# flag for input: 1 -- DWT coefficients; 0 -- FS PMF profiles
dwt_input = 1
#
if dwt_input:
    out_dim = 224  # DWT coeff.
else:
    out_dim = 200   # FS PMF
#
## model constants
#
batch_size = 32
learning_rate = 1e-3
epochs = 400
drop_rate = 0.3
lossThresh = 0.1
rangeThresh = 0.15
Nsplits = 20
Ntr_inst = 20
#
#
## paths
#
refDir = './data/TrainTestPaths/'
if dwt_input:
    resDir = './data/Test_Results/DWT_DNN/'
else:
    resDir = './data/Test_Results/PMF_DNN/'
fDatName = 'FSpmf_TrainTestPaths_'
fResName = 'TestPerf_df'
#
#
##  functions
#
def load_data(fPaths):
    FSpdfs = []
    for fPath in fPaths:
        with open(fPath, 'rb') as handle:
            FSpdf = pickle.load(handle)
        FSpdfs.append(FSpdf)
    #
    return np.stack(FSpdfs, axis=0)
#
# function for creating datasets of DWT coeffs concatenated in 1D vector
def dec_dwt1D(X_signal, wavelet='db5', n_max=3):
    X = []
    for i in range(X_signal.shape[0]):
        signal = X_signal[i, :]
        coeff1D = np.array([])
        coeff_list = pywt.wavedec(signal, wavelet=wavelet, level=n_max)
        for coeff in coeff_list:
            coeff1D = np.concatenate((coeff1D, coeff))
        X.append(coeff1D)
    X = np.stack(X, axis=0)
    #
    return X
#
# split X_test in neg and pos parts
def split_Test_neg_pos(X_test, Y_test):
    X_test_neg = X_test[~np.bool_(Y_test)]
    X_test_neg = X_test_neg.reshape(X_test_neg.shape[0], 1, X_test_neg.shape[1])
    #
    X_test_pos = X_test[np.bool_(Y_test)]
    X_test_pos = X_test_pos.reshape(X_test_pos.shape[0], 1, X_test_pos.shape[1])
    #
    Y_test_neg = np.zeros(X_test_neg.shape[0])
    Y_test_pos = np.ones(X_test_pos.shape[0])
    #
    return X_test_neg, X_test_pos, Y_test_neg, Y_test_pos
#
# function to define DNN model
def DNN_model(out_dim, drop_rate, learning_rate):
    # define DNN
    model = tf.keras.Sequential()
    model.add(layers.Dense(64, activation='leaky_relu', input_shape=[1, out_dim]))
    model.add(layers.Dropout(drop_rate))
    #
    model.add(layers.Dense(32, activation='leaky_relu'))
    model.add(layers.Dropout(drop_rate))
    #
    model.add(layers.Dense(16, activation='leaky_relu'))
    model.add(layers.Dropout(drop_rate))
    #
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    #
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    #
    return (model, cross_entropy, optimizer)
#
# function to create test dfs data for sample IDs to estimate test performance in terms of original samples
def make_test_dfs(fpathsTst, labelsTst):
    #
    sid_heal_df = pd.DataFrame(np.empty((len(labelsTst)-sum(labelsTst), 5)), columns=['FCid', 'Sid', 'Subid', 'ExSid', 'Pred'])
    sid_canc_df = pd.DataFrame(np.empty((sum(labelsTst), 5)), columns=['FCid', 'Sid', 'Subid', 'ExSid', 'Pred'])
    #
    i_heal = 0
    i_canc = 0
    for fPath in fpathsTst:
        path = fPath.split('_')
        cohort = path[0].split('/')[-1]
        FCid = path[1].split('/')[0]
        Sid = path[1].split('/')[-1]
        if len(path) == 3:
            ExSid = ''
            Subid = path[2].split('.')[0]
        elif len(path) == 4:
            ExSid = '_' + path[2]
            Subid = path[3].split('.')[0]
        else:
            ExSid = '_' + path[2] + '_' + path[3]
            Subid = path[4].split('.')[0]
        if cohort == 'healthy':
            sid_heal_df.iloc[i_heal, :4] = [FCid, Sid, Subid, ExSid]
            i_heal += 1
        else:
            sid_canc_df.iloc[i_canc, :4] = [FCid, Sid, Subid, ExSid]
            i_canc += 1
    return sid_heal_df, sid_canc_df
#
# function to update performance df with predictions
def update_df_pred(df, model, X_test):
    prob_pred = model(X_test, training=False)
    y_pred = np.int32(prob_pred > 0.5)[:, 0]
    df['Pred'] = y_pred
    return df
#
# function for test data accuracy
def testAccSpecSensAUCaucPR(X_test_neg, X_test_pos, Y_test_neg, Y_test_pos):
    #
    test_accuracy = tf.keras.metrics.BinaryAccuracy()
    test_accuracy.update_state(Y_test_neg, model(X_test_neg, training=False))
    test_spec = test_accuracy.result() * 100
    test_accuracy = tf.keras.metrics.BinaryAccuracy()
    test_accuracy.update_state(Y_test_pos, model(X_test_pos, training=False))
    test_sens = test_accuracy.result() * 100
    test_acc = test_spec * len(Y_test_neg) / (len(Y_test_neg) + len(Y_test_pos)) +\
               test_sens * len(Y_test_pos) / (len(Y_test_neg) + len(Y_test_pos))
    test_AUC = tf.keras.metrics.AUC() # ROC AUC
    test_AUC.update_state(np.concatenate((Y_test_neg, Y_test_pos)),\
                          model(np.concatenate((X_test_neg, X_test_pos)), training=False))
    test_AUCPR = tf.keras.metrics.AUC(curve='PR', name='pr_auc') # PR AUC
    test_AUCPR.update_state(np.concatenate((Y_test_neg, Y_test_pos)),\
                          model(np.concatenate((X_test_neg, X_test_pos)), training=False))
    return np.round(test_acc, 3), np.round(test_spec, 3), np.round(test_sens, 3), np.round(test_AUC.result(), 5),\
           np.round(test_AUCPR.result(), 5)
#
def testAccSpecSensAUCaucPR_indivSamples(sid_heal_df, sid_canc_df):
    nSer = sid_heal_df['Sid'].value_counts() # ground truth negatives
    N = len(nSer)
    fpSer = sid_heal_df.loc[sid_heal_df['Pred'] == 1, 'Sid'].value_counts() # candidate false positives
    FP = 0
    # loop through fp candidates and select only those that exceed 50% of their subsamples number
    for candFP in fpSer.index:
        if fpSer[candFP] / nSer[candFP] > 0.5:
            FP += 1
    # loop through all gr.truth negatives and mark FP those that exceed 50% of their subsamples number
    indivPred = []
    for negSID in nSer.index:
        if negSID in fpSer.index:
            if fpSer[negSID] / nSer[negSID] > 0.5:
                indivPred.append(1)
            else:
                indivPred.append(0)
        else:
            indivPred.append(0)
    #
    pSer = sid_canc_df['Sid'].value_counts() # ground truth positives
    P = len(pSer)
    fnSer = sid_canc_df.loc[sid_canc_df['Pred'] == 0, 'Sid'].value_counts() # candidate false negatives
    FN = 0
    # loop through fn candidates and select only those that exceed 50% of their subsamples
    for candFN in fnSer.index:
        if fnSer[candFN] / pSer[candFN] > 0.5:
            FN += 1
    # loop through all gr.truth positives and mark FN those that exceed 50% of their subsamples number
    for posSID in pSer.index:
        if posSID in fnSer.index:
            if fnSer[posSID] / pSer[posSID] > 0.5:
                indivPred.append(0)
            else:
                indivPred.append(1)
        else:
            indivPred.append(1)
    #
    indivLabl = np.concatenate((np.zeros(N, dtype=int), np.ones(P, dtype=int)))
    #
    Sp = (N - FP) / N * 100 # Specificity
    #
    Sn = (P - FN) / P * 100 # Sensitivity
    #
    Ac = (N - FP + P - FN) / (N + P) * 100 # Accuracy
    #
    AUC = tf.keras.metrics.AUC()
    AUC.update_state(indivLabl, np.array(indivPred)) # ROC AUC
    #
    AUCPR = tf.keras.metrics.AUC(curve='PR', name='pr_auc')
    AUCPR.update_state(indivLabl, np.array(indivPred)) # PR AUC
    #
    return np.round(Ac, 3), np.round(Sp, 3), np.round(Sn, 3), np.round(AUC.result(), 5), np.round(AUCPR.result(), 5)
#
def testAccSpecSensAUCaucPR_indivSamples_cons(sid_heal_df, sid_canc_df):
    nSer = sid_heal_df['Sid'].value_counts() # ground truth negatives
    N = len(nSer) # number of ground truth negatives
    fpSer = sid_heal_df.loc[sid_heal_df['Pred'] == 1, 'Sid'].value_counts() # candidate false positives
    FP = len(fpSer) # number of false positives
    # loop through all gr.truth negatives and mark FP those that have at least one such subsample
    indivPred = []
    for negSID in nSer.index:
        if negSID in fpSer.index:
            indivPred.append(1)
        else:
            indivPred.append(0)
    #
    pSer = sid_canc_df['Sid'].value_counts() # ground truth positives
    P = len(pSer) # number of ground truth positives
    fnSer = sid_canc_df.loc[sid_canc_df['Pred'] == 0, 'Sid'].value_counts() # candidate false negatives
    FN = len(fnSer) # number of false negatives
    # loop through all gr.truth positives and mark FN those that have at least one such subsample
    for posSID in pSer.index:
        if posSID in fnSer.index:
            indivPred.append(0)
        else:
            indivPred.append(1)
    #
    indivLabl = np.concatenate((np.zeros(N, dtype=int), np.ones(P, dtype=int)))    
    #
    Sn = (P - FN) / P * 100 # Sensitivity
    Sp = (N - FP) / N * 100 # Specificity
    Ac = (N - FP + P - FN) / (N + P) * 100 # Accuracy
    #
    AUC = tf.keras.metrics.AUC()
    AUC.update_state(indivLabl, np.array(indivPred)) # ROC AUC
    #
    AUCPR = tf.keras.metrics.AUC(curve='PR', name='pr_auc')
    AUCPR.update_state(indivLabl, np.array(indivPred)) # PR AUC
    #
    return np.round(Ac, 3), np.round(Sp, 3), np.round(Sn, 3), np.round(AUC.result(), 5), np.round(AUCPR.result(), 5)
#
#
##
# Training and testing in a loop over different train/test splits and with different initializations of DNN
#
time000 = time.time()
for v in range(1, (Nsplits + 1)):
    #
    # load reduced subsampling train/test set paths and load FS pmf profiles in the 51 - 250 bp range
    vers = str(v).zfill(2)
    with open(refDir + fDatName + vers + '.pickle', 'rb') as handle:
        fpathsTrn, labelsTrn, fpathsTst, labelsTst = pickle.load(handle)
    #
    # form training and testing datasets of DWT coeffs
    if dwt_input:
        X = dec_dwt1D(load_data(fpathsTrn))
        X_test = dec_dwt1D(load_data(fpathsTst))
    else:
        X = load_data(fpathsTrn)
        X_test = load_data(fpathsTst)    
    # labels
    Y = np.array(labelsTrn)
    Y_test = np.array(labelsTst)
    #
    # split X(Y)_test in neg and pos parts and reshape
    X_test_neg, X_test_pos, Y_test_neg, Y_test_pos = split_Test_neg_pos(X_test, Y_test)
    #
    # trim, reshape, normalize, shuffle and batch the train dataset
    buffer_size = X.shape[0] - (X.shape[0] % batch_size)
    X = X[:buffer_size, :]
    Y = Y[:buffer_size]
    X = X.reshape(buffer_size, 1, X.shape[1])
    X = tf.data.Dataset.from_tensor_slices(X).shuffle(buffer_size, seed=buffer_size).batch(batch_size)
    Y = tf.data.Dataset.from_tensor_slices(Y).shuffle(buffer_size, seed=buffer_size).batch(batch_size)
    #
    # performance df
    perf_df = pd.DataFrame(np.empty((Ntr_inst, 20)), index=np.arange(Ntr_inst),\
              columns=['AccSub', 'SpcSub', 'SnsSub', 'AucSub', 'AucPRSub',\
                       'AccMaj', 'SpcMaj', 'SnsMaj', 'AucMaj', 'AucPRMaj',\
                       'AccCon', 'SpcCon', 'SnsCon', 'AucCon', 'AucPRCon',\
                       'LossMean', 'LossRange', 'TrAccMean', 'TrEpochs', 'TrTime'])
    # test dfs
    sid_heal_df, sid_canc_df = make_test_dfs(fpathsTst, labelsTst)
    #
    # train a model with same hyper-parameters multiple times to assess variability due to model internal stochasticity
    time00 = time.time()
    for itr in range(Ntr_inst):
        # define DNN
        model, cross_entropy, optimizer = DNN_model(out_dim, drop_rate, learning_rate)
        # function for train/loss; needs to be defined for each new model run
        @tf.function
        def train_step(x, y):
            #
            with tf.GradientTape() as tape:
                y_hat = model(x, training=True)
                loss = cross_entropy(y, y_hat)
            #
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            #
            return loss
        #
        time0 = time.time()
        train_loss = []
        train_accuracy = []
        # loop over training epochs
        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.BinaryAccuracy()
            # training loop - using batches of batch_size
            for x, y in zip(X, Y):
                # train model
                loss = train_step(x, y)
                # batch loss
                epoch_loss_avg.update_state(loss)
                # batch accuracy
                epoch_accuracy.update_state(y, model(x, training=True))
            #
            train_loss.append(np.round(epoch_loss_avg.result(), 3))
            train_accuracy.append(np.round(epoch_accuracy.result() * 100, 3))
            # terminate training if the conditions are met
            if (epoch >= 4):
                lossMean = np.mean(train_loss[-5:])
                lossRange = np.max(train_loss[-5:]) - np.min(train_loss[-5:])
                TrAccMean = np.round(np.mean(train_accuracy[-5:]), 3)
                if (lossMean <= lossThresh) & (lossRange / lossMean <= rangeThresh):
                    break
        trTime = np.round(time.time() - time0, 2)
        # update dfs with predictions
        sid_heal_df = update_df_pred(sid_heal_df, model, X_test_neg)
        sid_canc_df = update_df_pred(sid_canc_df, model, X_test_pos)    
        # record test metrics
        perf_df.iloc[itr, 0:5] = testAccSpecSensAUCaucPR(X_test_neg, X_test_pos, Y_test_neg, Y_test_pos) # subsample performance
        perf_df.iloc[itr, 5:10] = testAccSpecSensAUCaucPR_indivSamples(sid_heal_df, sid_canc_df) # ind.samp perf, maj-of-vote
        perf_df.iloc[itr, 10:15] = testAccSpecSensAUCaucPR_indivSamples_cons(sid_heal_df, sid_canc_df) # ind.samp perf,consensus
        perf_df.iloc[itr, 15:] = (lossMean, lossRange, TrAccMean, epoch + 1, trTime)
        #
        if itr % 5 == 0:
            print("Tr.inst.: {} Tr.epochs: {} Mean Loss: {:.3f} Loss Range: {:.3f}  Mean Acc: {:.2f}%  Tr.Time: {} sec".format(\
                                                itr+1, epoch+1, lossMean, lossRange, np.round(TrAccMean, 3), trTime))
    #
    print('Split {}:  {} training instances completed in {} min'.format(v, Ntr_inst, np.round((time.time() - time00) / 60, 2)))
    print('   ')
    #
    # save perf_df for each data split on the go
    with open(resDir + fResName + vers + '.pickle', 'wb') as handle:
        pickle.dump(perf_df, handle, protocol=pickle.HIGHEST_PROTOCOL)    
#
print('All {} completed in {} hrs'.format(Nsplits, np.round((time.time() - time000) / 3600, 3)))