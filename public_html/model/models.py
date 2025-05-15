'''
Created on Sep 18, 2024

@author: jacek
'''
import tensorflow as tf
from keras import layers
import keras
import os
import numpy as np
from scipy import stats
from keras.models import load_model
import matplotlib.pyplot as plt
import muspy 
from keras.utils import to_categorical

# batch_size = 16
log_dir = "./exp/logs/"
sample_dir = './exp/samples/'
checkpoint_dir = "./exp/checkpoints/"

num_LSTM = 512
# latentDim = 20
# latentDim = 8
latentDim = 2

@tf.keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""  #

    def __init__(self, **kwargs):
        super(Sampling, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config()
        return config

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



def define_encoder(my_tensor_onehot, my_tensor_y_onehot_train):    
    # num_LSTM = 1024
    n_in = my_tensor_onehot.shape[1]
    n_feat = my_tensor_onehot.shape[2]
    
    
    # Define encoder input shape
    encoderInput2 = keras.Input(shape = (n_in, n_feat), name="music_sequence")
    encoded2 = encoderInput2
    # encoded2 = normalizer_layer(encoded2)   # JG normalizacja danych Wej
    
    y_labels = keras.Input(shape=(my_tensor_y_onehot_train.shape[1],),  name="emotion_labels") # JG dodanie label
    x = y_labels 
    # x = normalizer_layer_y(x)
    x = layers.Dense(n_in * my_tensor_y_onehot_train.shape[1])(x)
    x = layers.Reshape((n_in, my_tensor_y_onehot_train.shape[1]))(x)
    
    encoded2 = layers.concatenate([encoded2, x])  # JG połączenie tensorów  music_sequence plus labels
    
    
    # Define dense encoding layer connecting input to latent vector
    # encoded2 = layers.GRU(num_LSTM, activation='tanh', return_sequences=False)(encoderInput2)
    encoded2 = layers.GRU(num_LSTM, return_sequences=False)(encoded2)
    # encoded2 = layers.GRU(num_LSTM, return_sequences=False)(encoded2)
    
    encoded2 = layers.Dense(100, activation='relu')(encoded2)
    
    # Add mean and log variance layers.   activation='linear'
    z_mean = layers.Dense(latentDim,  name="z_mean")(encoded2)
    z_log_var = layers.Dense(latentDim,  name="z_log_var")(encoded2)
    # Add sampling layer.
    # z = layers.Lambda(sampling, output_shape=(latentDim,), name = "z_sampling")([z_mean, z_log_var])
    z = Sampling(name='Sampling_layer')([z_mean, z_log_var])
    
    # z = Sampling()([z_mean, z_log_var])
    encoder2 = keras.Model( [encoderInput2, y_labels], [z_mean, z_log_var, z], name = 'encoder')
    encoder2.summary()
    # Vizualizacja struktury
    from keras.utils import plot_model
    plot_model(encoder2, to_file='CVAE-class-encoder.png', show_shapes=True)
    
    return encoder2

def define_decoder(my_tensor_onehot, my_tensor_y_onehot_train):
    n_in = my_tensor_onehot.shape[1]
    n_feat = my_tensor_onehot.shape[2]
    # Define decoder input shape
    latent2 = keras.Input(shape = (latentDim,), name='z_sampling')
    
    y_labels = keras.Input(shape=(my_tensor_y_onehot_train.shape[1],),  name="emotion_labels") # JG dodanie label
   
    x = layers.concatenate([latent2, y_labels])
    
    decoded2 = layers.RepeatVector(n_in)(x)
    
    decoded2 = layers.GRU(num_LSTM, return_sequences=True)(decoded2)
    # decoded2 = layers.GRU(num_LSTM, return_sequences=True)(decoded2)
    # decoded2 = layers.LSTM(num_LSTM, activation='relu', return_sequences=True)(decoded2)
    
    # decoded2 = layers.TimeDistributed( layers.Dense(n_feat, activation="sigmoid") )(decoded2)
    decoded2 = layers.TimeDistributed( layers.Dense(n_feat, activation="softmax") )(decoded2)
    # activation="sigmoid"
    # # # Define the decoder models
    
    decoder2 = keras.Model([latent2, y_labels], decoded2, name='decoder')
    decoder2.summary()
    
    from keras.utils import plot_model
    plot_model(decoder2, to_file='CVAE-class-decoder.png', show_shapes=True)
    
    return decoder2

class CVAE(keras.Model): 
    def __init__(self, my_tensor_onehot, my_tensor_y_onehot_train, batch_size,  **kwargs):
        super(CVAE, self).__init__(  **kwargs)        
        self.encoder = define_encoder(my_tensor_onehot, my_tensor_y_onehot_train)
        self.decoder = define_decoder(my_tensor_onehot, my_tensor_y_onehot_train)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(   name="reconstruction_loss"    )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss") 

        self.val_total_loss_tracker = keras.metrics.Mean(name="val_total_loss")
        self.EmoSpace_reg_A_loss_tracker = keras.metrics.Mean(name="EmoSpace_A_reg_loss") 
        self.EmoSpace_reg_V_loss_tracker = keras.metrics.Mean(name="EmoSpace_V_reg_loss") 
        # Prepare the metrics.
        self.train_acc_metric = keras.metrics.CategoricalAccuracy()  # keras.metrics.SparseCategoricalAccuracy()
        self.val_acc_metric = keras.metrics.CategoricalAccuracy()

        self.checkpoint_dir = checkpoint_dir

        self.batch_size = batch_size
        self.VA_values = tf.Variable(tf.zeros( (self.batch_size , 2), tf.float32 ), trainable=False)         # (4, 2)
        # self.X_out = tf.Variable(tf.zeros( ( self.batch_size , self.batch_size ), tf.float32), trainable=False)  # (4, 4)


    @property
    def metrics(self):
        # resetowane po każdej epoce JG
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.train_acc_metric,

            self.val_total_loss_tracker,
            self.val_acc_metric,
            self.EmoSpace_reg_A_loss_tracker,
            self.EmoSpace_reg_V_loss_tracker,
        ]   
    def compute_reconstruction_loss(self, sequence_data, reconstruction):
        reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    # keras.losses.binary_crossentropy(sequence_data, reconstruction),  # pierwszy wariant
                    # axis=(1, 2),
                    keras.losses.categorical_crossentropy(sequence_data, reconstruction),  # drugi loss categorical_crossentropy -  lepiej dziala
                )    )
        return reconstruction_loss
        
    def compute_kl_loss(self, z_mean, z_log_var):
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum( kl_loss, axis=1))
        return kl_loss        
    
    @tf.function
    def train_step(self, data):
        sequence_data = data[0][0]
        labels = data[0][1]
        
        # tf.print("labels.shape", labels.shape)
        labels_1_batch = self.jg_convert_label_vector_to_label_number(labels)  # konvercja na numery labels [0, 1, 2, 3]
        # tf.print("len(labels_1_batch)", len(labels_1_batch))
        VA_values = self.jg_convert_labels_to_VA_values(labels_1_batch) # konvercja label numerow na VA values 
        # tf.print("len(labels): ", len(labels) )
        A_values = VA_values[:len(labels), 1]   # pobranie Arousal
        V_values = VA_values[:len(labels), 0]   # pobranie Valence
        # tf.print("A_values: ", A_values) 
        
        with tf.GradientTape() as tape:           
            
            # z_mean, z_log_var, z = self.encoder(data)
            z_mean, z_log_var, z = self.encoder([sequence_data, labels])
            # reconstruction = self.decoder(z)
            reconstruction = self.decoder([z, labels])
            
            #********** JG emo error   START
            Z_1 = z_mean[:, 1]   # Oś Y pionowa - Regulacja na osi PIONOWEJ Y
            # tf.print("Z_1: ", Z_1) 
            Z_0 = z_mean[:, 0]   # Oś X pozioma - Regulacja na osi poziomej X
            
             # ********************* cosine_similarity 
            # EmoSpace_reg_loss_A = self.current_beta.read_value()  * 100 * keras.losses.cosine_similarity (Z_1, A_values)
            # EmoSpace_reg_loss_V = self.current_beta.read_value()  * 100 * keras.losses.cosine_similarity (Z_0, V_values)
            # 30 dla_tylko_V czy tylko_A, 80-70 A-V
            EmoSpace_reg_loss_A = 80 * keras.losses.cosine_similarity (Z_1, A_values) # 100 *, 50, 80, 85
            EmoSpace_reg_loss_V = 70 * keras.losses.cosine_similarity (Z_0, V_values)   # 70 * 45, 70, 
            #********** JG emo error   END

            reconstruction_loss = self.compute_reconstruction_loss(sequence_data, reconstruction)
            kl_loss = self.compute_kl_loss(z_mean, z_log_var)            
            
            # total_loss = reconstruction_loss + kl_loss    # bez EmoReg
            
            # total_loss = reconstruction_loss + kl_loss + EmoSpace_reg_loss_A 
            # total_loss = reconstruction_loss + kl_loss + EmoSpace_reg_loss_V 
            total_loss = reconstruction_loss + kl_loss + EmoSpace_reg_loss_V + EmoSpace_reg_loss_A   # z EmoReg
            
            
        grads = tape.gradient(total_loss, self.trainable_weights)        
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        self.EmoSpace_reg_A_loss_tracker.update_state(EmoSpace_reg_loss_A)
        self.EmoSpace_reg_V_loss_tracker.update_state(EmoSpace_reg_loss_V)

          # Update training metric.
        # tf.print("sequence_data.shape:", sequence_data.shape)
        # tf.print("reconstruction.shape:", reconstruction.shape)
        self.train_acc_metric.update_state(sequence_data, reconstruction)     
        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "train_acc": self.train_acc_metric.result(), 
            "Emo_loss_A": self.EmoSpace_reg_A_loss_tracker.result(),
            "Emo_loss_V": self.EmoSpace_reg_V_loss_tracker.result(),           
        }
        
    def test_step(self, data):        
        sequence_data = data[0][0]
        labels = data[0][1]
        z_mean, z_log_var, z = self.encoder([sequence_data, labels])
        # reconstruction = self.decoder(z)
        reconstruction = self.decoder([z, labels])

        reconstruction_loss = self.compute_reconstruction_loss(sequence_data, reconstruction)        
        kl_loss = self.compute_kl_loss(z_mean, z_log_var)
        total_loss = reconstruction_loss + kl_loss

        self.val_total_loss_tracker.update_state(total_loss)
        self.val_acc_metric.update_state(sequence_data, reconstruction)       
        return {
            # "loss": self.total_loss_tracker.result(),
            # "recon_loss": self.reconstruction_loss_tracker.result(),
            # "kl_loss": self.kl_loss_tracker.result(),
            "loss": self.val_total_loss_tracker.result(),
            "acc": self.val_acc_metric.result(),            
        }   

    def call(self, inputs):
        tf.print("*** Jestem w call")  # JG 
        
        # print("*** UWAGA **** call --> inputs.shape: ", inputs.shape)
        y_pianoroll = inputs[0]
        y_labels = inputs[1]
        
        z_mean, z_log_var, z = self.encoder(inputs)
        x = self.decoder([z, inputs[1]])     
        return x 

    #******************************************
    # pobiera label numer z vektora
    # konvercja na numery labels [0, 1, 2, 3]
    #******************************************
    def jg_convert_label_vector_to_label_number(self, train_y_onehot_1_batch):
        # labels_1_batch = np.argmax(train_y_onehot_1_batch, axis=1)
        labels_1_batch = tf.math.argmax(train_y_onehot_1_batch, axis=1)
        # print("labels_1_batch: ", labels_1_batch)
        return labels_1_batch

    #******************************************
    # Konwertuje label number to  VA_values [4,2]  [: ,Valence, Arousal]
    # zapisuje wyniki w self.VA_values = tf.Variable
    #******************************************
    # @tf.function
    def jg_convert_labels_to_VA_values(self, labels_1_batch):
        # tf.print("labels_1_batch: ", labels_1_batch)
        batch_size = len(labels_1_batch)
        # batch_size = self.batch_size
        # VA_values = np.zeros( (batch_size, 2) )
        # VA_values = tf.Variable(tf.zeros( (batch_size, 2), tf.float32 ))
        pos_in_space = 1
        # pos_in_space = 0.5        
        
        for i in range(batch_size):           
            if labels_1_batch[i] == 0 : # e1
                # VA_values[i] = [pos_in_space, pos_in_space]
                self.VA_values[i].assign( [pos_in_space, pos_in_space] )
            elif labels_1_batch[i] == 1 :  # e2
                # VA_values[i] = [-pos_in_space, pos_in_space]
                self.VA_values[i].assign( [-pos_in_space, pos_in_space] )
            elif labels_1_batch[i] == 2 :  # e3
                # VA_values[i] = [-pos_in_space, -pos_in_space]
                self.VA_values[i].assign( [-pos_in_space, -pos_in_space] )
            elif labels_1_batch[i] == 3 :  # e4
                # VA_values[i] = [pos_in_space, -pos_in_space]
                self.VA_values[i].assign( [pos_in_space, -pos_in_space] )
            else:
                print ("Błąd tworzenia Distance VA_values")    
            print(i, "\t label: ", labels_1_batch[i], "\t VA_values[i]: ", self.VA_values[i]) 
            
        # print(VA_values)
        # tf.print("self.VA_values: \n", self.VA_values)   
        return self.VA_values
    
    #******************************************
    # Konwertuje label number to  VA_values [4,2]  [: ,Valence, Arousal]
    # zapisuje wyniki w zwyklej array   użyty do wyliczenia SCC
    #******************************************
    def jg_convert_labels_to_VA_values_array (self, labels):
        # tf.print("labels_1_batch: ", labels_1_batch)
        batch_size = len(labels)
        VA_values = np.zeros( (batch_size, 2) )
        pos_in_space = 1
        # pos_in_space = 0.5        
        
        for i in range(batch_size):           
            if labels[i] == 0 : # e1
                VA_values[i] = [pos_in_space, pos_in_space]
            elif labels[i] == 1 :  # e2
                VA_values[i] = [-pos_in_space, pos_in_space]
            elif labels[i] == 2 :  # e3
                VA_values[i] = [-pos_in_space, -pos_in_space]
            elif labels[i] == 3 :  # e4
                VA_values[i] = [pos_in_space, -pos_in_space]
            else:
                print ("Błąd tworzenia Distance VA_values")    
            # print(i, "\t label: ", labels[i], "\t VA_values[i]: ", VA_values[i]) 
            
        return VA_values

    def save_model(self, epoch):            
        filename1 = "decoder_epoch_%04d_.h5" % (epoch)
        path = os.path.join(self.checkpoint_dir, filename1) 
        self.decoder.save(path)
        print('%s saved to: \n%s' % (filename1, path))   
        
        filename1 = "encoder_epoch_%04d_.h5" % (epoch) 
        path = os.path.join(self.checkpoint_dir, filename1) 
        self.encoder.save(path) 
        print('%s saved to: %s' % (filename1, path))
        
    def load_model(self, epoch): 
        filename1 = "decoder_epoch_%04d_.h5" % (epoch)
        path = os.path.join(self.checkpoint_dir, filename1) 
        self.decoder = load_model(path)
        print('%s was loaded from %s' % (filename1, path))        
       
        filename1 = "encoder_epoch_%04d_.h5" % (epoch)
        path = os.path.join(self.checkpoint_dir, filename1) 
        self.encoder = load_model(path) 
        print('%s was loaded from %s' % (filename1, path))
        
    #*********************************************
    # liczy liczbe punktów w ćwiartkach
    # zwraca liste i array lista_elementow_array
    #*********************************************
    def jg_policz_punkty_w_cwiartkach (self, z_mean):
        lista_elementow = []
        
        for i in range (z_mean.shape[0]):
            element = z_mean[i]
            if element[0] >= 0 and element[1] >= 0 :  # VA  emo = 0 
                lista_elementow.append (0)
            elif element[0] < 0 and element[1] >= 0 :   # VA  emo = 1 
                lista_elementow.append (1)
            elif element[0] < 0 and element[1] < 0 :  # VA  emo = 2 
                lista_elementow.append (2)
            elif element[0] >= 0 and element[1] < 0 :  # VA  emo = 3 
                lista_elementow.append (3)
            else   :
                print ("Bład liczenie elementow in quarters AV")   
    
        unique_elements, counts_elements = np.unique(lista_elementow, return_counts=True)
        print("Frequency of unique values of the said array:")
        lista_elementow_array = np.asarray((unique_elements, counts_elements))
        print(lista_elementow_array)
        print("Suma elementów: ", np.sum(lista_elementow_array[1]) )    
        return lista_elementow , lista_elementow_array    
    
    #*********************************************
    # liczy liczbe trafień w ćwiartkach - czyli punkt jest w ćwiartce tej co jego etykieta train_y
    # zwraca array liczba_trafien_quarter_array
    #*********************************************
    def jg_policz_trafienia_w_cwiartkach(self, z_mean, lista_elementow, train_y):
        licznik_trafien = []
        liczba_trafien_quarter  = []
        for i in range (z_mean.shape[0]):
            if lista_elementow[i] == train_y[i]:
                licznik_trafien.append(1)
                liczba_trafien_quarter.append(train_y[i])
        print("len(licznik_trafien) ", len(licznik_trafien) )
        print("len(liczba_trafien_quarter) ", len(liczba_trafien_quarter) )
    
        unique_elements, counts_elements = np.unique(liczba_trafien_quarter, return_counts=True)
        print("Frequency of unique values of the said array:")
        liczba_trafien_quarter_array = np.asarray((unique_elements, counts_elements))
        print(liczba_trafien_quarter_array)
        print("Suma elementów liczba_trafien_quarter_array[1]: ", np.sum(liczba_trafien_quarter_array[1]) )    
        return liczba_trafien_quarter_array   
    
    #*********************************************
    # liczy liczbe trafień w ćwiartkach - czyli punkt jest w ćwiartce tej co jego etykieta train_y
    # zwraca array = liczba_trafien_half_up_down_array
    #*********************************************
    def jg_policz_trafienia_w_polowkach_up_down(self, z_mean, lista_elementow, train_y):
        liczba_trafien_half_up_down = []    
        
        for i in range (z_mean.shape[0]):
            if (lista_elementow[i] == 0 or lista_elementow[i] == 1) and (train_y[i] == 0  or train_y[i] == 1) :  # up half AV plane            
                liczba_trafien_half_up_down.append(0)  # 0 = up
            elif ( lista_elementow[i] == 2 or lista_elementow[i] == 3 ) and (train_y[i] == 2  or train_y[i] == 3) : # down half AV plane            
                liczba_trafien_half_up_down.append(1)  # 1 = down            
    
        print("len(liczba_trafien_half_up_down) ", len(liczba_trafien_half_up_down) )
        
        unique_elements, counts_elements  = np.unique(liczba_trafien_half_up_down, return_counts=True)
        print("Frequency of unique values of the said array:")
        liczba_trafien_half_up_down_array = np.asarray((unique_elements, counts_elements ))
        print(liczba_trafien_half_up_down_array)
        # print(liczba_trafien_half_up_down_array.shape)
        # print("Suma elementów: ", np.sum(count_output_half) )
        return liczba_trafien_half_up_down_array 

        
    #**********************************************************
    # ### Funkcja rysowania Latent Space, print and loging mean and std
    #**********************************************************
    def plot_latent_space(self, my_tensor_onehot, my_tensor_y_onehot_train, epochs, plt_show_on = False):
        # display a 2D plot of the digit classes in the latent space
        z_mean, _, _ = self.encoder.predict([my_tensor_onehot, my_tensor_y_onehot_train])    
        osX = 0
        osY = 1
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        # fig, (ax2) = plt.subplots(1, 1, figsize=(5,5))
        plt.figure(figsize=(5, 5))
        labels = np.argmax(my_tensor_y_onehot_train, axis = 1 )
       
        group = labels
        cdict = {0: 'black', 1: 'red', 2: 'blue', 3: 'green'}  # kolorowy plot ze znakami
        # cdict = {0: 'black', 1: 'black', 2: 'black', 3: 'black'} # czarno-bialy plot ze znakami
        marker_dict = {0: '+', 1: 'v', 2: 'x', 3: 'o'}  # czarno-bialy plot ze znakami
        
        emo = {0: 'e1', 1: 'e2', 2: 'e3', 3: 'e4'}
        
        for g in np.unique(group):
            ix = np.where(group == g)
            # plt.scatter(z_mean[ix, osX], z_mean[ix, osY], c = cdict[g], label = emo[g], s = 30)
            
            plt.scatter(z_mean[ix, osX], z_mean[ix, osY], c = cdict[g], 
                        marker = marker_dict[g], label = emo[g], s = 30) # czarno-bialy plot ze znakami
            
        # ax2.set_xlabel ("z["+ str(osX) + "]")      
        # ax2.set_ylabel ("z["+ str(osY) + "]")
        # plt.title("Latent space") 
        plt.xlabel ("z["+ str(osX) + "]")      
        plt.ylabel ("z["+ str(osY) + "]")
        
        
        
        plt.legend()
        zakres = 6  # 3
        plt.xlim(-zakres, zakres) # Set the range of x-axis        
        plt.ylim(-zakres, zakres) # Set the range of y-axis
        plt.grid(True)  
        
        plt.text(zakres/2, zakres-0.8, 'Q1', fontsize = 'large', fontweight = 'bold')
        plt.text(-zakres/2, zakres-0.8, 'Q2', fontsize = 'large', fontweight = 'bold')
        plt.text(-zakres/2, -zakres+0.5, 'Q3', fontsize = 'large', fontweight = 'bold')
        plt.text(zakres/2, -zakres+0.5, 'Q4', fontsize = 'large', fontweight = 'bold')
        
        #**** OSIE START
        xpoints = np.array([-zakres, zakres])
        ypoints = np.array([0, 0])
        plt.plot(xpoints, ypoints, linestyle = '-', color="black")   
        
        xpoints = np.array([0, 0]) 
        ypoints = np.array([-zakres, zakres])
        plt.plot(xpoints, ypoints, linestyle = '-', color="black")  
        #**** OSIE END
        
    
        #****************************** save plot to file - START
        epoch = epochs
        
        filename1 = 'latent_space_%04d.png' % (epoch)
        path = os.path.join(log_dir, filename1) 
        plt.savefig(path)
        print('%s was saved to %s' % (filename1, path))
        #****************************** save plot to file - END
        if(plt_show_on):
            plt.show()
        plt.close()
        
        
        #********************* Policz Spearman correlation coefficient ro - START 
        #********************* liczy SCC między  labels i miejscami punktów w  LatentSpace
        print ("*************************************************************************")
        print ("Spearman rank-order correlation coefficient ro między labels i osiami w LatentSpace" )
        print ("*************************************************************************")
        VA_values = self.jg_convert_labels_to_VA_values_array(labels)
        # print("Debug VA_values[:10] ",  VA_values[:10] ) 
        # print("Debug z_mean[:10] ",  z_mean[:10] ) 
         
         
        a = VA_values[:, 1]
        b = z_mean[:, 1]
        res = stats.spearmanr(a, b)
        Spearman_Ro_A = res.statistic
        print( "Spearman_Ro_A:", "{:.4f}".format(Spearman_Ro_A)) 
               
        a = VA_values[:, 0]
        b = z_mean[:, 0]
        res = stats.spearmanr(a, b)
        Spearman_Ro_V = res.statistic
        print( "Spearman_Ro_V:", "{:.4f}".format(Spearman_Ro_V))         
             
        print( "np.mean([Spearman_Ro_A, Spearman_Ro_V]) ",  "{:.4f}".format(np.mean([Spearman_Ro_A, Spearman_Ro_V])) )        
        print ("********************************************")
        #********************* Policz Spearman correlation coefficient - END 
        
        #********************* Policz punkty w ćwiartkach AV - START *****************************************
        print ("********************************************")
        print ("Parametry trafień w AV Plane po regulacji EmoReg" )
        print ("********************************************")
        lista_elementow , lista_elementow_array = self.jg_policz_punkty_w_cwiartkach (z_mean)
        liczba_trafien_quarter_array = self.jg_policz_trafienia_w_cwiartkach(z_mean, lista_elementow, labels)
        procent_trafien_quarter = np.sum(liczba_trafien_quarter_array[1]) / len(lista_elementow) 
        print("procent_trafien_quarter: ",  "{:.3f}".format(procent_trafien_quarter)  ) 
        
        procent_trafien_per_quarter = liczba_trafien_quarter_array[1,:] / lista_elementow_array[1,:]
        print("e1 e2 e3 e4 % = ",  procent_trafien_per_quarter )     
        
        
        
        liczba_trafien_half_up_down_array = self.jg_policz_trafienia_w_polowkach_up_down(z_mean, lista_elementow, labels)         
        procent_trafien_half_up_down = np.sum(liczba_trafien_half_up_down_array[1])  / len(lista_elementow) 
        print("procent_trafien_half_up_down: ", "{:.3f}".format(procent_trafien_half_up_down)  )  
        
        print ("********************************************")
        
        #********************* Policz punkty w ćwiartkach AV - END *****************************************
    
        #*************************** Write Latent_space.log START
        filename = 'Latent_space.log'
        filepath = os.path.join(log_dir, filename)
        with open(filepath, 'a') as f:
            f.write ("********Parametry Latent space********************* \n") 
            f.write ("Epoch {:d} \n".format(epochs)) 
            f.write ("*************************************************** \n") 
            
            print ("z_mean.shape", z_mean.shape)    
            for j in range(0, z_mean.shape[1]):
                a = z_mean[:,j]
        #         print (a.shape)
                # print (j, "mean: ", np.mean(a), "std: ", np.std(a) )  
                print ("{:d},  mean: {:.4f},  std: {:.4f} ".format(j, np.mean(a), np.std(a)))
                f.write ("{:d},  mean: {:.4f},  std: {:.4f} \n".format(j, np.mean(a), np.std(a)))
                
            f.write ("***************** Liczba trafien plane AV ********************************** \n")  
            f.write ( "liczba_trafien_quarter_array \n" )
            content = str(liczba_trafien_quarter_array)
            f.write(content)  
            f.write ( "\n" )  
            f.write ("Procent_trafien_quarter: {:.3f} \n".format(procent_trafien_quarter) )
            f.write ("e1 e2 e3 e4 % = {} \n".format( procent_trafien_per_quarter) )
            
            f.write ( "liczba_trafien_half_up_down_array \n" )
            content = str(liczba_trafien_half_up_down_array)
            f.write(content)  
            f.write ( "\n" )             
            f.write ("Procent_trafien_half_up_down: {:.3f} \n ".format(procent_trafien_half_up_down)  )  
            
            f.write ("*** Spearman rank-order correlation coefficient między labels [-1,1] i osiami w LatentSpace ***\n")  
            f.write ("Spearman_Ro_A:: {:.4f} \n".format(Spearman_Ro_A)  ) 
            f.write ("Spearman_Ro_V:: {:.4f} \n".format(Spearman_Ro_V)  ) 
            f.write ("Mean {:.4f} \n".format( np.mean([Spearman_Ro_A, Spearman_Ro_V])  ) )
        #*************************** Write Latent_space.log END   
        return procent_trafien_quarter  # zwraca proc traf 
     
    #*****************************************************************
    # funkcja jg_print_score konweruje sekwencję liczb(pitch_rep_new)
    # na muspy misic, # pokazuje też nuty show_score
    #*****************************************************************
    def jg_print_score(self, result, intToNote, show_score_on = True ):
        # pitch_rep_new = [np.argmax(i) for i in my_tensor_onehot[1, :]]
        pitch_rep_new = [intToNote[np.argmax(i)] for i in result ]   # wersja ze słownikiem intToNote
        # print(pitch_rep_new)
        my_pitch_rep_one = np.array (pitch_rep_new)
    #     my_pitch_rep_one   
        # powrot do music z pitch_representation
        my_new_music = muspy.from_pitch_representation(my_pitch_rep_one, resolution=4, 
                                        program=0,  use_hold_state=True, default_velocity=90)
        # my_new_music.adjust_resolution(target=1024) 
    #     print(my_new_music)
        # muspy.write_midi('./new_generated_5.mid', my_new_music)
        
        # if (show_score_on):
        scoreplotter_Jg = my_new_music.show_score(figsize=(15,1))
        return my_new_music, scoreplotter_Jg
    
    #*****************************************************************
    # Funkcja generująca NOWE sekwencje generatedSequence_onehot o danej emocji emo={0, 1, 2, 3}
    # losowanie ze środków ćwiartek
    #*****************************************************************
    def jg_generate_seq_emo(self, emo, EmoReg_on = False):
        # Generate chords from randomly generated latent vector
        scale=1
        scale=2
        randomVec = np.random.normal(scale=scale, size=(1,latentDim))
        print('randomVec: ', randomVec)
        # print('randomVec.shape: ', randomVec.shape)
        if (EmoReg_on):            
            shift_vector = np.zeros( (1, 2) )   
            # shift_value = 3.0
            # shift_value = 2.0
            shift_value = 2.0
            if  (emo == 0): 
                shift_vector = [[shift_value, shift_value]]
            elif (emo == 1):
                shift_vector = [[-shift_value, shift_value]]
            elif (emo == 2):
                shift_vector = [[-shift_value, -shift_value]]
            elif (emo == 3):
                shift_vector = [[shift_value, -shift_value]]
              
            randomVec = randomVec  + shift_vector             # Przesynięcie vektora losowego AV wg emocji 
            print('New randomVec after shifting: ', randomVec)
    
        cond_num = emo 
        condition_emo = to_categorical(cond_num, 4).reshape(1,-1)
        # print('condition_emo: ', condition_emo)
        # print('condition_emo.shape: ', condition_emo.shape)
    
        generatedSequence_onehot = self.decoder([randomVec, condition_emo])
        print ('generatedChords.shape: ', generatedSequence_onehot.shape)
        return generatedSequence_onehot
    
    #*****************************************************************
    # funkcja generująca 10 sampli o różnych emocjacj emo={0, 1, 2, 3} i zapisująca je do katalogu
    #*****************************************************************
    def jg_run_sampler_v1(self, epoch, intToNote):
        num_samples = 10    
        
        for i in range(num_samples):
            ind = (i % 4)        
            print(ind, end =" ")
            emo = ind
            generatedSequence_onehot = self.jg_generate_seq_emo(emo)
            my_new_music , scoreplotter_Jg = self.jg_print_score(generatedSequence_onehot[0, :], intToNote)
    
            filename1 = 'MIDI_epoch_%04d_%s.mid' % (epoch, i)
            path = os.path.join(sample_dir, filename1)
            muspy.write_midi(path , my_new_music) 
            print('MIDI saved to %s' % path)
            
    #*****************************************************************
    # funkcja generująca 10 sampli o różnych emocjacj emo={0, 1, 2, 3} 
    # i zapisująca je do katalogu + zapisuje nuty
    #*****************************************************************
    def jg_run_sampler_v2(self, epoch, intToNote, EmoReg_on = False):
        num_samples = 10
    
        for i in range(num_samples):
            ind = (i % 4)        
            print(ind, end =" ")
            emo = ind
            generatedSequence_onehot = self.jg_generate_seq_emo(emo, EmoReg_on)
            my_new_music , scoreplotter_Jg = self.jg_print_score(generatedSequence_onehot[0, :], intToNote)
    
            filename1 = 'MIDI_epoch_%04d_%s.mid' % (epoch, i)
            path = os.path.join(sample_dir, filename1)
            muspy.write_midi(path , my_new_music) 
            print('MIDI saved to %s' % path)
    
            fig_jg = scoreplotter_Jg.fig
    
            filename1 = 'Samples_%s_images_%04d.png' % (epoch, i )
            path = os.path.join(sample_dir , filename1)
            print (path)
            fig_jg.savefig(path)
            print('%s was saved to %s' % (filename1, path))
        
        plt.close() 
    
if __name__ == '__main__':
    print("Test module: models")
    
     
    # vae = CVAE(MODEL_CONFIG) 