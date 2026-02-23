#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import random
import IPython
import csv

import math
import user_results as ur
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

from sklearn.model_selection import train_test_split

import sys

import netCDF4 as nc


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as patches


print('tensorflow version:', tf.__version__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


in_file = sys.argv[1]


# In[2]:


def extract_data(data, features):
    x_data = []
    for i in features:
        x_data.append(data[i])
    x_data = np.array(x_data).transpose()
    y_data = data['obs']
    return x_data, np.array(y_data)
 
def get_dataset():

    ifile = in_file
    idata = nc.Dataset(ifile)
    input_data = idata['MDL_VAR'][:]

    Features = ['jday','b01','b02','b03','b04','b05','b06','b07','b08','b09','b10','b11','b12','b13','b14','b15','b16','sza','saa','vza','vaa','lon','lat']
    n_features = len(Features)
    x_input = input_data[:,0:23]
    y_input = input_data[:,23]

    x_train, x_tmp,  y_train, y_tmp  = train_test_split(x_input, y_input, test_size=0.3, shuffle=True, random_state=1004)
    x_valid, x_test, y_valid, y_test = train_test_split(x_tmp,   y_tmp,   test_size=0.33, shuffle=True, random_state=1004)
    print('dataset x shapes =', x_train.shape, x_valid.shape, x_test.shape)
    print('dataset y shapes =', y_train.shape, y_valid.shape, y_test.shape)
    
    total = np.concatenate([x_train,x_valid,x_test]) # Merge (To get Total avg and std for standardization of inputs)
    for i in range(0,n_features):
        mean = np.mean(total[:,i])
        std  = np.std(total[:,i])
        x_train[:,i] = (x_train[:,i]-mean)/std
        x_valid[:,i] = (x_valid[:,i]-mean)/std
        x_test[:,i]  = (x_test[:,i]-mean)/std


    total = np.concatenate([y_train,y_valid,y_test]) # Merge (To get Total avg and std for standardization of inputs)
    mean = np.mean(total)
    std  = np.std(total)

    y_train = (y_train-mean)/std
    y_valid = (y_valid-mean)/std
    y_test  = (y_test-mean)/std

    return x_train, y_train, x_valid, y_valid, x_test, y_test


# In[5]:


def create_model_dnn(hp_parameters):
    hp_dr, hp_lr, hp_act, hp_opt, hp_K_init, hp1_dense = hp_parameters # Setting hyperparameters
    inputx = keras.Input(shape=[n_features,])
    x = inputx
    for i in range(len(hp1_dense)):
        x = keras.layers.Dense(hp1_dense[i], kernel_initializer=hp_K_init)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(hp_act)(x)
        x = keras.layers.Dropout(hp_dr)(x)
    x = keras.layers.Dense(1)(x)
    model = keras.models.Model(inputx, x)
    if hp_opt == 'RMSprop': optimizer = keras.optimizers.RMSprop(learning_rate=hp_lr)
    if hp_opt == 'adam': optimizer = keras.optimizers.Adam(learning_rate=hp_lr)
    model.compile(optimizer=hp_opt, loss='mse')    
    return model

# In[6]:

def create_model_conv1d(hp_parameters):
    hp_dr, hp_lr, hp_act, hp_opt, hp_K_init, hp1_conv = hp_parameters # Setting hyperparameters
    inputx = keras.Input(shape=[n_features,])
    x = keras.layers.Reshape([n_features,1])(inputx) # 2차원 shape 변경; CNN 또는 RNN (LSTM)은 2차원 shape 필요
    for i in range(len(hp1_conv)): # len(hp1_conv); 파라미터 개수 (loop 수행을 위해)
        x = keras.layers.Conv1D(hp1_conv[i], kernel_size=2, kernel_initializer=hp_K_init)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(hp_act)(x)
        x = keras.layers.MaxPool1D(2)(x)
        x = keras.layers.Dropout(hp_dr)(x)
    x = keras.layers.GlobalMaxPool1D()(x)
    x = keras.layers.Dense(1)(x)
    model = keras.models.Model(inputx, x)
    if hp_opt == 'RMSprop': optimizer = keras.optimizers.RMSprop(learning_rate=hp_lr)
    if hp_opt == 'adam': optimizer = keras.optimizers.Adam(learning_rate=hp_lr)
    model.compile(optimizer=hp_opt, loss='mse')    
    return model

# In[7]:


# Get hyperparameters

# 히든 유닛을 레이어에 따라 감소하는 방향으로
def get_unit_dn(n_layers, Min=128, Max=512, Step=32, subStep=16):
    hp_unit = random.choice(np.arange(Min, Max+1, Step))
    for i in range(1, n_layers):    
        try:
            calc = np.arange(int(hp_unit[i-1]/2), hp_unit[i-1]+1, subStep)
        except:
            calc = np.arange(int(hp_unit/2), hp_unit+1, subStep)
        c_calc = random.choice(calc)
        hp_unit = np.vstack([hp_unit, c_calc])
    hp_unit = hp_unit.reshape(-1)
    return hp_unit.tolist()

# 히든 유닛을 레이어에 따라 증가시키는 방향으로
def get_unit_up(n_layers, Min=64, Max=128, Step=32, subStep=16):
    hp_unit = random.choice(np.arange(Min, Max+1, Step))
    for i in range(1, n_layers):    
        try:
            calc = np.arange(hp_unit[i-1], int(hp_unit[i-1]*1.5)+1 , subStep)
        except:
            calc = np.arange(hp_unit, int(hp_unit*1.5)+1 , subStep)
        c_calc = random.choice(calc)
        hp_unit = np.vstack([hp_unit, c_calc])
    hp_unit = hp_unit.reshape(-1)
    return hp_unit.tolist()

# 히든 유닛을 랜덤으로
def get_unit_rn(n_layers, Min=64, Max=256, Step=16):
    hp_units = random.choice(np.arange(Min, Max+1, Step))
    for i in range(1, n_layers):    
        hp_init  = random.choice(np.arange(Min, Max+1, Step))
        hp_units = np.vstack([hp_units, hp_init])
    hp_units = hp_units.reshape(-1)
    return hp_units.tolist()

def get_parameters_dnn():
    hp_dr   = random.choice([0.2,0.3,0.4,0.5])
    hp_lr   = random.choice([0.003,0.005,0.001])
    hp_act  = random.choice(['relu', 'selu'])
    hp_opt  = random.choice(['RMSprop', 'adam'])
    hp_K_init = random.choice(['GlorotNormal', 'GlorotUniform', 'he_normal', 'he_uniform'])
    hp1_numlayer = random.choice(np.arange(3,7))
    #hp1_dense = get_unit_up(hp1_numlayer, Min=64, Max=256, Step=32, subStep=16)
    hp1_dense = get_unit_dn(hp1_numlayer, Min=128, Max=512, Step=32, subStep=16)
    
    print(hp_dr, hp_lr, hp_act, hp_opt, hp_K_init, hp1_dense)
    return hp_dr, hp_lr, hp_act, hp_opt, hp_K_init, hp1_dense

def get_parameters_conv1d():
    
    hp_dr   = random.choice([0.2,0.3,0.4,0.5])
    hp_lr   = random.choice([0.003,0.005,0.001])
    hp_act  = random.choice(['relu', 'selu'])
    hp_opt  = random.choice(['RMSprop', 'adam'])
    hp_K_init = random.choice(['GlorotNormal', 'GlorotUniform', 'he_normal', 'he_uniform'])
    hp1_numlayer = random.choice(np.arange(1,4))
    #hp1_conv = get_unit_up(hp1_numlayer, Min=64, Max=256, Step=32, subStep=16)
    hp1_conv = get_unit_dn(hp1_numlayer, Min=128, Max=512, Step=32, subStep=16)

    print(hp_dr, hp_lr, hp_act, hp_opt, hp_K_init, hp1_conv)
    return hp_dr, hp_lr, hp_act, hp_opt, hp_K_init, hp1_conv


# Best model save
def Check_point(save_path, model_name, model_num, vbose=1):
    try: os.mkdir(save_path)
    except: pass
    file_path  = save_path+'model_'+model_name+'_'+str(model_num).zfill(3)+'.hdf5'
    check_point = ModelCheckpoint(filepath=file_path, monitor='val_loss', 
                                  verbose=vbose, save_best_only=True, mode='auto', save_weights_only=False)
    return check_point

# 더이상 모델 성능에 진전이 없을 경우 stop
def Early_stopping(patience_epoch=30):
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience_epoch)
    return early_stopping

# learning rate 조절
def Step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.9 # Reduction rate
    epochs_drop = 10               # (밑수, 지수)
    lrate = initial_lrate*math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def Learning_schedule(vbose=1):
    learning_schedule = LearningRateScheduler(Step_decay, verbose=vbose)
    return learning_schedule
  
def model_run(m_path, m_name, model, m_index, n_epoch, n_batch, n_patience):
    Save_path  = m_path+m_name+'/check_points/'
    class ClearOutput(keras.callbacks.Callback):
        def on_epoch_end(*args, **kwargs):
            IPython.display.clear_output(wait=True)
            print('processing number = ',now_num)
            
    checkpoint        = Check_point(save_path=Save_path, model_name= m_name, model_num=m_index)
    learning_schedule = Learning_schedule()
    earlystopping     = Early_stopping(patience_epoch=n_patience)
    callback_list     = [checkpoint, earlystopping, ClearOutput()]# 남은 callback 함수: learning_schedule

    hist = model.fit(x_train, y_train, batch_size=n_batch,
                    epochs=n_epoch, verbose=1, validation_data=(x_valid,y_valid),
                    callbacks=callback_list)


    # --- START Model Performance ---

    loss_fl = m_path+m_name+'/'+m_name+'_'+str(m_index).zfill(3)+'_loss.txt'
    grph_fl = m_path+m_name+'/'+m_name+'_'+str(m_index).zfill(3)+'_loss.png'


    # --- Write Model Performance ---

    with open(loss_fl,'w') as lossfile:
        for trn_loss, val_loss in zip(hist.history['loss'], hist.history['val_loss']):
            lossfile.write('Training_'+str(trn_loss)+'_Validation_'+str(val_loss)+'\n')


    # --- Plot Model Performance ---

    font_path_bold    = '/data1/home/kari/.local/share/fonts/helvetica/Helvetica_Bold.ttf'
    font_path_regular = '/data1/home/kari/.local/share/fonts/helvetica/Helvetica.ttf'
    font_prop_bold    = fm.FontProperties(fname=font_path_bold   , size=14)
    font_prop_regular = fm.FontProperties(fname=font_path_regular, size=14)

    fig = plt.figure(facecolor='#232b38')
    ax  = fig.add_subplot(111, facecolor='#232b38')

    ax.plot(hist.history['loss']    , label='Training'  , color='#008ffb', linewidth=2)
    ax.plot(hist.history['val_loss'], label='Validation', color='#fe9b19', linewidth=2)

    for spine in ax.spines.values():
        spine.set_edgecolor('#C2C2DB')

    ax.tick_params(axis='x', colors='#C2C2DB')
    ax.tick_params(axis='y', colors='#C2C2DB')

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop_regular)
        label.set_color('#C2C2DB')

    ax.set_xlabel('Epoch', color='#C2C2DB', fontproperties=font_prop_bold)
    ax.set_ylabel('Loss' , color='#C2C2DB', fontproperties=font_prop_bold)
    ax.set_ylim([0,0.8])

    legend = plt.legend(loc="upper right", frameon=False, prop=font_prop_bold)
    legend.get_texts()[0].set_color('#C2C2DB')
    legend.get_texts()[1].set_color('#C2C2DB')

    plt.gca().add_patch(
        patches.Rectangle(
            (0, 0), 1, 1, transform=fig.transFigure,
            linewidth=0.5, edgecolor='#49DCCF', facecolor='none', zorder=10 ))

    plt.tight_layout()

    plt.savefig(grph_fl, dpi=300)
    print(grph_fl)

    # --- END Model Performance ---

    
    new_model = keras.models.load_model(Save_path+'model_'+m_name+'_'+str(m_index).zfill(3)+'.hdf5') # callback에서 저장된 best model 호출
    pred_trn  = new_model.predict(x_train, batch_size=n_batch, verbose=1)
    pred_val  = new_model.predict(x_valid, batch_size=n_batch, verbose=1)
    pred_test = new_model.predict(x_test, batch_size=n_batch, verbose=1)
    pred_trn  = pred_trn.reshape(-1) # predict 실행후 나온 결과는 2차원 shape, reshape이용 1차원으로 shape 형태 변경
    pred_val  = pred_val.reshape(-1)
    pred_test = pred_test.reshape(-1)
    trn_r2, trn_rmse = ur.calc_evaluation(x_train, y_train, pred_trn)
    val_r2, val_rmse = ur.calc_evaluation(x_valid, y_valid, pred_val)
    test_r2, test_rmse = ur.calc_evaluation(x_test, y_test, pred_test)
    optEP = len(hist.history['val_loss'])-n_patience

    return trn_r2, trn_rmse**2, val_r2, val_rmse**2, test_r2, test_rmse**2, optEP, pred_trn, pred_val, pred_test

# In[18]:


def main(m_name, m_path, n_epoch, n_batch, n_patience, n_repeat, header=False):
#===================================
#결과 파일 csv로 생성
#===================================
    oDir = m_path+m_name+'/'
    try:
        print('make ', oDir)
        os.makedirs(oDir)
    except:
        pass
    ofile  = (oDir+m_name+'_results.csv')
    fcheck = os.path.isfile(ofile)
    if not fcheck:
        f = open(ofile, mode='w', newline='')
        wr = csv.writer(f)
        if not header:
            wr.writerow(['num', 'trn_r2', 'trn_mse', 'val_r2', 'val_mse', 'test_r2','test_mse','optimal_epoch', 'n_batch'])
        else:
            wr.writerow(['num', 'trn_r2', 'trn_mse', 'val_r2', 'val_mse', 'test_r2','test_mse','optimal_epoch', 'n_batch']+header)
        f.close()
#==================================
# 결과 csv 파일로부터 번호를 얻기 위해
#==================================
    pd_data = pd.read_csv(ofile)
    first_num = len(pd_data) 
#==================================
    print(m_name)
    for i in range(first_num, n_repeat):
        global now_num
        now_num = i
        if m_name == 'dnn':
            hp_parameters = get_parameters_dnn()
            model = create_model_dnn(hp_parameters)
        else:
            hp_parameters = get_parameters_conv1d()
            model = create_model_conv1d(hp_parameters)
        trn_r2, trn_mse, val_r2, val_mse, test_r2, test_mse, optEpoch, pred_trn, pred_val, pred_test = model_run(m_path, m_name, model, i, n_epoch, n_batch, n_patience)
        f  = open(ofile, mode='a', newline='')
        wr = csv.writer(f)
        if not header:
            wr.writerow([i, trn_r2, trn_mse, val_r2, val_mse, test_r2, test_mse, optEpoch, n_batch])
        else:
            wr.writerow([i, trn_r2, trn_mse, val_r2, val_mse, test_r2, test_mse, optEpoch, n_batch]+list(hp_parameters))
        f.close()

        out_name = m_path+m_name+'/model_'+m_name+'_'+str(i).zfill(3)
        ur.plot_results(x_train, y_train, pred_trn, x_lim=[0,1], y_lim=[0,1], pname='Train', ofig=out_name+'_train.png')
        ur.plot_results(x_valid, y_valid, pred_val, x_lim=[0,1], y_lim=[0,1], pname='Valid', ofig=out_name+'_valid.png')
        ur.plot_results(x_test, y_test, pred_test, x_lim=[0,1], y_lim=[0,1], pname='Test', ofig=out_name+'_test.png')
        out_pred_data = np.savez(out_name+'_result_datset.npz', x_train=y_train, y_train=pred_trn, x_valid=y_valid, y_valid=pred_val, x_test=y_test, y_test=pred_test)

# In[19]:


# Call dataset
x_train, y_train, x_valid, y_valid, x_test, y_test = get_dataset()
n_features = x_train.shape[1]

# In[20]:


m_path = './callbacks/'
n_epoch = 300
n_patience = 10
n_repeat   = 30
headers = ['hp_dr', 'hp_lr', 'hp_act', 'hp_opt', 'hp_K_init', 'hp1_dense']


n_batch = 256

m_name = 'conv1d'
main(m_name, m_path, n_epoch, n_batch, n_patience, n_repeat, header=headers)

n_batch = 256

m_name = 'dnn'
main(m_name, m_path, n_epoch, n_batch, n_patience, n_repeat, header=headers)


