'''
Created on Sep 18, 2024

@author: jacek
'''
# import os
# ####*IMPORANT*: Have to do this line *before* importing tensorflow
# os.environ['PYTHONHASHSEED']=str(0)

from numpy import load
import numpy as np
import os
import time
from datetime import timedelta
import random
import matplotlib.pyplot as plt

import tensorflow as tf
# from tensorflow import keras
import keras
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from tensorflow.keras import layers
from keras import layers
from keras.utils import to_categorical
# from tensorflow.keras import Model
# from tensorflow.keras.layers import  LSTM, Dense, RepeatVector, TimeDistributed

#********************** SET Random seed START ********************************
os.environ["KERAS_BACKEND"] = "tensorflow"

# os.environ['PYTHONHASHSEED'] = str(33)
# np.random.seed(33)
# random.seed(33)
# tf.random.set_seed(33)

# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) backend random seed
# 3) `python` random seed
keras.utils.set_random_seed(33)

# If using TensorFlow, this will make GPU ops as deterministic as possible,
# but it will affect the overall performance, so be mindful of that.
tf.config.experimental.enable_op_determinism()
#************************* SET Random seed END* ****************************

from model.models import CVAE

log_dir = "./exp/logs/"

#***********************************************
# nowa wersja kodowania oneHot ze słownikami noteToInt
#***********************************************
def jg_makeTensorOneHot_v2 (res, uniqueNotes, noteToInt):
    
    print("*********** Kodowanie oneHot ze słownikami noteToInt ")
    dict_len = len(uniqueNotes)       # liczba nut w słowniku                   
    batch_len = res.shape[0]
    timeline_len = res.shape[1]

    x = np.zeros((batch_len, timeline_len, dict_len), dtype=bool)     # wartości 0-129
    x.shape
    for i in range (0, batch_len):
        for t in range (0, timeline_len):        
            val =  int(res[i, t])
            val = noteToInt[val]    # konwersja ntu na numer
            x[i, t, val] = 1
    print("Result tensor shape: ", x.shape) 
    return x

#************************************************
# Ładowanie danych z my_tensor_344.npz
#************************************************
def load_data_1voice_emo():
    ## Ładowanie my_tensor (bez onehot) i etykiety my_tensor_y 
    print("*********** Ładowanie my_tensor (bez onehot) i etykiety my_tensor_y ")
    dict_data = load('my_tensor_344.npz')
    my_tensor = dict_data['arr_0']
    print("my_tensor.shape", my_tensor.shape)
    
    dict_data = load('my_tensor_y_344.npz')
    my_tensor_y = dict_data['arr_0']
    print("my_tensor_y.shape", my_tensor_y.shape)
    print("my_tensor_y.shape", my_tensor_y.shape)
    
    print(my_tensor_y.tolist())
    
    return my_tensor, my_tensor_y

#********************************************
### Tworzenie słowników noteToInt i intToNote
#********************************************
def create_Dictionaries (my_tensor):
   
    print("*********** Tworzenie słowników noteToInt i intToNote")
    # Tworzenie słowników noteToInt i intToNote
    orginalNotes = my_tensor.reshape(my_tensor.shape[0] * my_tensor.shape[1] * my_tensor.shape[2])
    print("orginalNotes.shape: ", orginalNotes.shape)
    uniqueNotes = np.unique([s for s in orginalNotes] )
    print(uniqueNotes)
    
    noteToInt = dict(zip(uniqueNotes, list(range(0, len(uniqueNotes)))))
    # noteToInt[129]
    intToNote = {i: c for c, i in noteToInt.items()}
    # intToNote[30]
    len(uniqueNotes)
    
    return noteToInt, intToNote, uniqueNotes
    

def podzial_zbioru_na_Test_Train(my_tensor_onehot, my_tensor_y):
    from sklearn.model_selection import train_test_split  
    X_train, X_test, Y_train, Y_test  = train_test_split(my_tensor_onehot, my_tensor_y, 
                                                     test_size=0.20, random_state=42, 
                                                     stratify=my_tensor_y,
                                                     #                                                      shuffle= False                                                     
                                                    )
    my_tensor_onehot = X_train
    my_tensor_onehot_val = X_test
    print ('my_tensor_onehot.shape', my_tensor_onehot.shape )
    print ('my_tensor_onehot_val.shape', my_tensor_onehot_val.shape )
    
    my_tensor_y_train = Y_train
    my_tensor_y_test = Y_test
    print ('my_tensor_y_train.shape', my_tensor_y_train.shape )
    print ('my_tensor_y_test.shape', my_tensor_y_test.shape )
    
    unique_elements, counts_elements = np.unique(my_tensor_y_train, return_counts=True)
    print("Frequency of unique values of the said array:")
    print(np.asarray((unique_elements, counts_elements)))
    
    unique_elements, counts_elements = np.unique(my_tensor_y_test, return_counts=True)
    print("Frequency of unique values of the said array:")
    print(np.asarray((unique_elements, counts_elements))) 
    
    return my_tensor_onehot, my_tensor_onehot_val, my_tensor_y_train, my_tensor_y_test

def labels_to_onehot_vectors(my_tensor_y_train, my_tensor_y_test):
    # conversion labels to onehot vectors
    print(my_tensor_y_train)
    # one-hot encoding
    my_tensor_y_onehot_train  = to_categorical(my_tensor_y_train) 
    my_tensor_y_onehot_test = to_categorical(my_tensor_y_test)
    print(my_tensor_y_onehot_train.shape, my_tensor_y_onehot_test.shape)
    print(my_tensor_y_onehot_train[0])
    
    return my_tensor_y_onehot_train, my_tensor_y_onehot_test

#***********************************************
# ### Funkcja rysuje Loss na przestrzeni epok
#***********************************************
def jg_show_loss(loss, reconstruction_loss, kl_loss, val_loss, plt_show_on=False): 
    epoki = range(len(loss))
    plt.figure()
    plt.plot(epoki, loss, 'ro', label='total_loss')
    plt.plot(epoki, val_loss, 'y', label='val_total_loss')
    plt.plot(epoki, reconstruction_loss, 'g', label='reconstruction_loss')
    plt.plot(epoki, kl_loss, 'b', label='kl_loss')
    plt.title('Loss')
    plt.legend()

    epoch = len(loss)
    filename1 = 'Loss_%04d.png' % ( epoch)
    path = os.path.join(log_dir, filename1)
    
    plt.savefig(path)
    print("*******************************************")
    print('Loss_%04d.png saved to %s' % (epoch, path))
    
    if (plt_show_on):
        plt.show()
    plt.close()

#***********************************************
# ### Funkcja rysuje Accuracy  na przestrzeni epok
#***********************************************    
def jg_show_acc(acc, val_acc, plt_show_on=False): 
    epoki = range(len(acc))
    plt.figure()
    plt.plot(epoki, acc, 'b', label='acc')
    plt.plot(epoki, val_acc, 'r', label='val_acc')
    plt.title('Accuracy')
    plt.legend()

    epoch = len(acc)
    filename1 = 'Accuracy_%04d.png' % ( epoch)
    path = os.path.join(log_dir, filename1)
    
    plt.savefig(path)
    print("*******************************************")
    print('Accuracy_%04d.png saved to %s' % (epoch, path))
    if(plt_show_on):
        plt.show()
    plt.close()

def jg_show_EmoSpace_reg_loss(loss, loss_V): 
    epoki = range(len(loss))

    plt.figure()

    plt.plot(epoki, loss, 'r', label='Emo_reg_A_loss')
    plt.plot(epoki, loss_V, 'g', label='Emo_reg_V_loss')
   

    plt.title('EmoSpace_reg_loss trenowania')
    plt.legend()

    epoch = len(loss)
    filename1 = 'Loss_EmoSpace_reg_loss_%04d.png' % (epoch)
    path = os.path.join( log_dir, filename1)
    
    plt.savefig(path)
    print("*******************************************")
    print('Loss_EmoSpace_reg_loss_%04d.png saved to %s' % (epoch, path))

    # plt.show()
    plt.close() 
    
def jg_show_procent_trafien_quarter(procent_trafien, epoch, epoch_log_step): 
    epoki = range(len(procent_trafien))
    epoki_arr = np.array(epoki)
    epoki_arr = (epoki_arr+1) * epoch_log_step
    # print ("epoki_arr", epoki_arr)

    plt.figure()

    # plt.plot(epoki_arr, procent_trafien, 'r', label='procent_trafien in quarter')
    plt.plot(epoki_arr, procent_trafien, color = '#000000')
   
    # plt.title('procent_trafien in quarter')
    # plt.legend()
    
    plt.ylim(0.0,1.0)
    plt.xlabel ("Epoka")      
    plt.ylabel ("Dokładność")

    # epoch = len(loss)
    filename1 = 'Procent_trafien_quarter_%04d.png' % (epoch)
    path = os.path.join( log_dir, filename1)
    
    plt.savefig(path)
    print("*******************************************")
    print('Procent_trafien_quarter_%04d.png saved to %s' % (epoch, path))

    # plt.show()
    plt.close()     
#***********************************************
# Główna pętla trenowania
#***********************************************
def main_train():
    training_start_time = time.time() 
    import os         
    # to keep using Keras 2, you can first install tf_keras, and then export the environment variable TF_USE_LEGACY_KERAS=1.
    # os.environ['TF_USE_LEGACY_KERAS'] = '1'     
    
    print("Tensorflow ver:",tf.__version__)
    print("Keras ver:", keras.__version__)    
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU
    print("GPU: ", tf.config.list_physical_devices('GPU'))
    print("*************************************************")   
    
    
    my_tensor, my_tensor_y = load_data_1voice_emo()      
    noteToInt, intToNote, uniqueNotes = create_Dictionaries (my_tensor)   
        
    my_tensor_onehot = jg_makeTensorOneHot_v2 (my_tensor, uniqueNotes, noteToInt)
    print ('my_tensor_onehot.shape', my_tensor_onehot.shape )
    print ('my_tensor_onehot[0,0]', my_tensor_onehot[0,0] )    
    my_tensor_onehot = my_tensor_onehot.astype('float32')   # jg zmiana typu 
    
    print("***********  Podział zbioru Test Train Split - Stratify")
    my_tensor_onehot, my_tensor_onehot_val, my_tensor_y_train, my_tensor_y_test = podzial_zbioru_na_Test_Train(my_tensor_onehot, my_tensor_y)
    print("******************** Conversion labels to onehot vectors")
    my_tensor_y_onehot_train, my_tensor_y_onehot_test = labels_to_onehot_vectors(my_tensor_y_train, my_tensor_y_test)


    # parametry do sieci
    seq_length = my_tensor_onehot.shape[1]
    dict_length = my_tensor_onehot.shape[2]   # wielkość słownika
    inputDim = my_tensor_onehot.shape[2] * my_tensor_onehot.shape[1]  # 130 * 64
    nSamples = my_tensor_onehot.shape[0]
    # Flatten sequence of chords into single dimension
    # my_tensor_onehot_Flat = my_tensor_onehot.reshape(nSamples, inputDim)
    # print("trainChordsFlat.shape: ", my_tensor_onehot_Flat.shape) 
    
    
    # encoder2 = define_encoder(my_tensor_onehot, my_tensor_y_onehot_train)
    # decoder2 = define_decoder(my_tensor_onehot, my_tensor_y_onehot_train)
    
    batch_size = 16
    
    # vae3 = CVAE(encoder2, decoder2, batch_size)
    vae3 = CVAE(my_tensor_onehot, my_tensor_y_onehot_train, batch_size)
    # optimizer = keras.optimizers.Adam(learning_rate=0.001)
    optimizer = keras.optimizers.RMSprop(learning_rate=0.001)   # JG ten dobry
    vae3.compile(optimizer = optimizer, metrics=["Accuracy"])
    
    
    #*************** Główna pętla trenowania START
    epoch = 700
    # epoch = 100
    # epoch = 30
    
    # history = vae3.fit([my_tensor_onehot, my_tensor_y_onehot_train], my_tensor_onehot, epochs=epoch, batch_size=batch_size , 
    #                     validation_data=([my_tensor_onehot_val, my_tensor_y_onehot_test], my_tensor_onehot_val),               )
    
    # Stworzenie na nowo plików Latent_space.log eval_samples.log   
    log_dir = "./exp/logs/" 
    filepath = 'Latent_space.log'
    filepath = os.path.join(log_dir, filepath)
    log_latent_space = open(filepath, 'w') 
    log_latent_space.close()
    
    loss_all = []
    reconstruction_loss_all = []
    kl_loss_all = []
    val_loss_all = []
    acc_all = []
    val_acc_all = []
    EmoSpace_reg_loss_all = []
    EmoSpace_reg_loss_Valence_all = []
    procent_trafien_quarter_all = []
    
    for epoch in range(epoch):
        print('{:-^80}'.format(' Epoch {} Start '.format(epoch + 1)))
        print("epoch: ", epoch+1)
    
        history = vae3.fit([my_tensor_onehot, my_tensor_y_onehot_train], my_tensor_onehot, epochs=1, batch_size=batch_size , 
                        validation_data=([my_tensor_onehot_val, my_tensor_y_onehot_test], my_tensor_onehot_val),
                      )
        loss_all.append(history.history['loss'][0])
        reconstruction_loss_all.append(history.history['recon_loss'][0])
        kl_loss_all.append(history.history['kl_loss'][0])
        val_loss_all.append(history.history['val_loss'][0])
        acc_all.append(history.history['train_acc'][0])
        val_acc_all.append(history.history['val_acc'][0])
        EmoSpace_reg_loss_all.append(history.history['Emo_loss_A'][0])
        EmoSpace_reg_loss_Valence_all.append(history.history['Emo_loss_V'][0])
        

        epoch_log_step =  50  # 10, 50
        if (epoch + 1) % epoch_log_step == 0:  
        # if (epoch + 1) % 50 == 0: 
            procent_trafien_quarter =  vae3.plot_latent_space( my_tensor_onehot, my_tensor_y_onehot_train,  epoch + 1)
            procent_trafien_quarter_all.append(procent_trafien_quarter)  # JG dodaje wartosc historii procent_trafien_quarter
            vae3.save_model(epoch + 1)            
            vae3.jg_run_sampler_v2(epoch + 1, intToNote, EmoReg_on = True)
            
        
          
    #*************** Główna pętla trenowania END  
    training_time = time.time() - training_start_time    
    print('{:-^80}'.format(' Training time '))
    print("time   {:8.2f} s".format(training_time)) 
    print("time formated: ", str(timedelta(seconds=training_time))) 
    
    jg_show_loss(loss_all, reconstruction_loss_all, kl_loss_all, val_loss_all) 
    jg_show_acc(acc_all, val_acc_all)
    jg_show_EmoSpace_reg_loss(EmoSpace_reg_loss_all, EmoSpace_reg_loss_Valence_all)
    jg_show_procent_trafien_quarter(procent_trafien_quarter_all, epoch+ 1, epoch_log_step)
    
    print('{:-^80}'.format(' End '))
    
def test_procent_trafien(): 
    
    procent_trafien_quarter_all = [] 
    procent_trafien_quarter_all.append(0.5)
   
    procent_trafien_quarter_all.append(0.55)
    procent_trafien_quarter_all.append(0.7)
    
    epoch = 30
    epoch_log_step = 10
    
     
    jg_show_procent_trafien_quarter(procent_trafien_quarter_all, epoch, epoch_log_step) 

if __name__ == '__main__':
    
    main_train()    # główne trenowanie modelu
    # test_procent_trafien()
    # pass

