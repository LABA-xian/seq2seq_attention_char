# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:21:16 2019

@author: user
"""

# Start by importing all the things we'll need.
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model, model_from_json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed, Dropout, LSTMCell, RNN,Bidirectional, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras import backend as K
from keras.callbacks import EarlyStopping
import unicodedata
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from keras.utils import plot_model

file_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'chainsea_all_API_3'))
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

class seq2seq_Attention_all():
    
    def __init__(self, weight_name, data_name):
        
        self.weight_name = weight_name
        self.data_name = data_name
        
        self.path_to_file = file_path + '/' + self.data_name
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        
        #num_examples = 118000 # Full example set.
    

        num_examples =  60000 # Partial set for faster training
        self.input_data, self.teacher_data, self.input_lang, self.target_lang, self.len_input, self.len_target = self.load_dataset(self.path_to_file, num_examples)
        
        
        self.target_data = [[self.teacher_data[n][i+1] for i in range(len(self.teacher_data[n])-1)] for n in range(len(self.teacher_data))]
        self.target_data = tf.keras.preprocessing.sequence.pad_sequences(self.target_data, maxlen=self.len_target, padding="post")
        self.target_data = self.target_data.reshape((self.target_data.shape[0], self.target_data.shape[1], 1))
        
        # Shuffle all of the data in unison. This training set has the longest (e.g. most complicated) data at the end,
        # so a simple Keras validation split will be problematic if not shuffled.
        p = np.random.permutation(len(self.input_data))
        self.input_data = self.input_data[p]
        self.teacher_data = self.teacher_data[p]
        self.target_data = self.target_data[p]
        
        
        BUFFER_SIZE = len(self.input_data)
        self.BATCH_SIZE = 32  # small data 64 big 32
        self.embedding_dim = 200 #small data 100 big 200
        self.units = 256 #small data 128 big 512
        self.vocab_in_size = len(self.input_lang.word2idx)
        self.vocab_out_size = len(self.target_lang.word2idx)
    

# Download the file



    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    def preprocess_sentence(self, w):
    #    w = unicode_to_ascii(w.lower().strip())
    #    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    #    w = re.sub(r'[" "]+', " ", w)
    #    w = re.sub(r"[a-zA-Z]+", " ", w)
    #    w = w.rstrip().strip()
        w = "<start> " + w + " <end>"
        return w

    def max_length(self, t):
        return max(len(i) for i in t)
    
    def create_dataset(self, path, num_examples):
        lines = open(path, encoding="UTF-8").read().strip().split("\n")
        random.shuffle(lines)
        word_pairs = [[self.preprocess_sentence(w) for w in l.split("\t")] for l in lines[:num_examples]]
        return word_pairs

    def load_dataset(self, path, num_examples):
        pairs = self.create_dataset(path, num_examples)
        out_lang = LanguageIndex(sp for en, sp in pairs)
        in_lang = LanguageIndex(en for en, sp in pairs)
        input_data = [[in_lang.word2idx[s] for s in en.split(' ')] for en, sp in pairs]                
                
        output_data = [[out_lang.word2idx[s] for s in sp.split(' ')] for en, sp in pairs]
    
        max_length_in, max_length_out = self.max_length(input_data), self.max_length(output_data)
        input_data = tf.keras.preprocessing.sequence.pad_sequences(input_data, maxlen=max_length_in, padding="post")
        output_data = tf.keras.preprocessing.sequence.pad_sequences(output_data, maxlen=max_length_out, padding="post")
        return input_data, output_data, in_lang, out_lang, max_length_in, max_length_out


    def train_test_split(self, x, t, y):
        
        train_data = x[:int(len(x) * 0.8)]
        teach_data = t[:int(len(t) * 0.8)]
        train_data_y = y[:int(len(y) * 0.8)]
        
        test_data = x[int(len(x) * 0.8):]
        test_teach_data = t[int(len(x) * 0.8):]
        test_y = y[int(len(x) * 0.8):]

        
        
        return train_data, teach_data, train_data_y, test_data, test_teach_data, test_y
        
        
        
    
    def train(self):
        # Encoder Layers
        attenc_inputs = Input(shape=(self.len_input,), name="encoder_attention_inputs")
        attenc_emb = Embedding(input_dim=self.vocab_in_size, output_dim=self.embedding_dim, name = 'encoder_attention_embedding')
        attenc_lstm = Bidirectional(LSTM(units=self.units, return_sequences=True, return_state=True), name = 'bi_encoder_attention_lstm')
        attenc_outputs, forward_h, forward_c, backward_h, backward_c = attenc_lstm(attenc_emb(attenc_inputs))
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]
        
#        attenc_outputs, attstate_h, attstate_c = attenc_lstm(attenc_emb(attenc_inputs))
#        attenc_states = [attstate_h, attstate_c]
        
        attdec_inputs = Input(shape=(None,), name="decoder_attention_inputs")
        attdec_emb = Embedding(input_dim=self.vocab_out_size, output_dim=self.embedding_dim, name = 'decoder_attention_embedding')
        attdec_lstm = LSTMWithAttention(units=self.units*2, return_sequences=True, return_state=True, name = 'decoder_attention_lstm')
        # Note that the only real difference here is that we are feeding attenc_outputs to the decoder now.
        # Nice and clean!
        attdec_lstm_out, _, _ = attdec_lstm(inputs=attdec_emb(attdec_inputs), 
                                            constants=attenc_outputs, 
                                            initial_state=encoder_states)
        
        attdec_d1 = Dense(self.vocab_out_size, activation="softmax", name = 'Dense1')
        attdec_out = attdec_d1((BatchNormalization(name = 'BN1')(attdec_lstm_out)))
        
        attmodel = Model([attenc_inputs, attdec_inputs], attdec_out)
        attmodel.compile(optimizer=tf.train.AdamOptimizer(), loss="sparse_categorical_crossentropy", metrics=['sparse_categorical_accuracy'])
        
        
        attmodel.summary()
        
        plot_model(attmodel, to_file='model.png', show_shapes=True)
        
#        atthist = attmodel.fit([self.input_data, self.teacher_data], self.target_data,
#                         batch_size=self.BATCH_SIZE,
#                         epochs=25,
#                         validation_split=0.2)
        
        
        train_data, teach_data, train_data_y, test_data, test_teach_data, test_y = self.train_test_split(self.input_data, self.teacher_data, self.target_data)
   
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=1)        
        model = attmodel.fit_generator(self.generate_batch_data_random(train_data, teach_data, train_data_y, self.BATCH_SIZE),
                               validation_data=self.generate_batch_data_random(test_data, test_teach_data, test_y, self.BATCH_SIZE),
                               steps_per_epoch=train_data.shape[0] // self.BATCH_SIZE,
                               validation_steps=test_data.shape[0] // self.BATCH_SIZE,
                               epochs=10,
                               callbacks=[early_stopping]
                               )
        
        attmodel.save_weights(self.weight_name)

        plt.plot(model.history['sparse_categorical_accuracy'])
        plt.plot(model.history['val_sparse_categorical_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()


        plt.plot(model.history['loss'])
        plt.plot(model.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()        
        
        return attmodel
    
    
    def create_model(self):
        # Encoder Layers
        attenc_inputs = Input(shape=(self.len_input,), name="encoder_attention_inputs")
        attenc_emb = Embedding(input_dim=self.vocab_in_size, output_dim=self.embedding_dim, name = 'encoder_attention_embedding')
        attenc_lstm = Bidirectional(LSTM(units=self.units, return_sequences=True, return_state=True), name = 'bi_encoder_attention_lstm')
        attenc_outputs, forward_h, forward_c, backward_h, backward_c = attenc_lstm(attenc_emb(attenc_inputs))
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]
        
#        attenc_outputs, attstate_h, attstate_c = attenc_lstm(attenc_emb(attenc_inputs))
#        attenc_states = [attstate_h, attstate_c]
        
        attdec_inputs = Input(shape=(None,), name="decoder_attention_inputs")
        attdec_emb = Embedding(input_dim=self.vocab_out_size, output_dim=self.embedding_dim, name = 'decoder_attention_embedding')
        attdec_lstm = LSTMWithAttention(units=self.units*2, return_sequences=True, return_state=True, name = 'decoder_attention_lstm')
        # Note that the only real difference here is that we are feeding attenc_outputs to the decoder now.
        # Nice and clean!
        attdec_lstm_out, _, _ = attdec_lstm(inputs=attdec_emb(attdec_inputs), 
                                            constants=attenc_outputs, 
                                            initial_state=encoder_states)
        
        attdec_d1 = Dense(self.vocab_out_size, activation="softmax", name = 'Dense1')
        attdec_out = attdec_d1((BatchNormalization(name = 'BN1')(attdec_lstm_out)))
        
        attmodel = Model([attenc_inputs, attdec_inputs], attdec_out)
        attmodel.compile(optimizer=tf.train.AdamOptimizer(), loss="sparse_categorical_crossentropy", metrics=['sparse_categorical_accuracy'])

        
        
        return attmodel    
        
    
    def createAttentionInference(self, model, attention_mode=False):
        # Create an inference model using the layers already trained above.
        
        
        model.load_weights(file_path + "/" + self.weight_name, by_name = True)
        
        global graph
        graph = tf.get_default_graph()    
        
        
        Attention_model = model
        
        attention_encoder_input = Input(shape=(self.len_input,))
        
        encoder_attention_embedding = Attention_model.get_layer('encoder_attention_embedding')(attention_encoder_input)
        
#        encoder_outs, encoder_h, encoder_c = Attention_model.get_layer('encoder_attention_lstm')(encoder_attention_embedding)

        encoder_outs, forward_h, forward_c, backward_h, backward_c = Attention_model.get_layer('bi_encoder_attention_lstm')(encoder_attention_embedding)
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])

        encoder_model = Model(attention_encoder_input, [encoder_outs, state_h, state_c])
        
#        encoder_model = Model(attention_encoder_input, [encoder_outs, encoder_h, encoder_c])
        
        
        
        attention_decoder_inputs = Input(shape=(None,))
        
        decoder_state_h = Input(shape=(self.units*2,))
        decoder_state_c = Input(shape=(self.units*2,))
        decoder_states_inputs = [decoder_state_h, decoder_state_c]        
        
        attenc_seq_out = Input(shape=encoder_outs.get_shape()[1:], name="attenc_seq_out")        

        decoder_attention_embedding = Attention_model.get_layer('decoder_attention_embedding')(attention_decoder_inputs)
        Attention_model.get_layer('decoder_attention_lstm').cell.setAttentionMode(attention_mode)
        decoder_out, decoder_h, decoder_c = Attention_model.get_layer('decoder_attention_lstm')(decoder_attention_embedding, initial_state=decoder_states_inputs, constants = attenc_seq_out)
        
        
        bn1 = Attention_model.get_layer('BN1')(decoder_out)
        
        dense1 = Attention_model.get_layer('Dense1')(bn1)
        
        decoder_model = Model(inputs=[attention_decoder_inputs, decoder_state_h, decoder_state_c,  attenc_seq_out], 
                              outputs=[dense1 , decoder_h, decoder_c])

        
        return encoder_model, decoder_model
    
        
#        attencoder_model = Model(attenc_inputs, [attenc_outputs, attstate_h, attstate_c])
#        state_input_h = Input(shape=(units,), name="state_input_h")
#        state_input_c = Input(shape=(units,), name="state_input_c")
#        attenc_seq_out = Input(shape=attenc_outputs.get_shape()[1:], name="attenc_seq_out")
#        inf_attdec_inputs = Input(shape=(None,), name="inf_attdec_inputs")
#        attdec_lstm.cell.setAttentionMode(attention_mode)
#        attdec_res, attdec_h, attdec_c = attdec_lstm(attdec_emb(inf_attdec_inputs), 
#                                                     initial_state=[state_input_h, state_input_c], 
#                                                     constants=attenc_seq_out)
#        attinf_model = None
#        if not attention_mode:
#            inf_attdec_out = attdec_d2(attdec_d1(attdec_res))
#            attinf_model = Model(inputs=[inf_attdec_inputs, state_input_h, state_input_c, attenc_seq_out], 
#                                 outputs=[inf_attdec_out, attdec_h, attdec_c])
#        else:
#            attinf_model = Model(inputs=[inf_attdec_inputs, state_input_h, state_input_c, attenc_seq_out], 
#                                 outputs=[attdec_res, attdec_h, attdec_c])
#        return attencoder_model, attinf_model
    
    
    def sentence_to_vector(self, sentence, lang):
        pre = self.preprocess_sentence(sentence)
        vec = np.zeros(self.len_input)
#        sentence_list = [lang.word2idx[s] for s in pre.split(' ')]


        sentence_list = []
        for s in  pre.split(' '):
            if s in lang.word2idx: 
                sentence_list.append(lang.word2idx[s])
            else:
                pass
        for i,w in enumerate(sentence_list):
            vec[i] = w
        return vec

    def sample(self, preds, temperature=1.5):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    # Given an input string, an encoder model (infenc_model) and a decoder model (infmodel),
    # return a translated string.
    def translate(self, input_sentence, infenc_model, infmodel, attention=False):
        
        with graph.as_default():
        
            input_sentence = input_sentence.replace(" ","")
            input_sentence = " ".join(input_sentence)
            sv = self.sentence_to_vector(input_sentence, self.input_lang)
            # Reshape so we can use the encoder model. New shape=[samples,sequence length]
            sv = sv.reshape(1,len(sv))
            [emb_out, sh, sc] = infenc_model.predict(x=sv)
            
            i = 0
            start_vec = self.target_lang.word2idx["<start>"]
            stop_vec = self.target_lang.word2idx["<end>"]
            # We will continuously feed cur_vec as an input into the decoder to produce the next word,
            # which will be assigned to cur_vec. Start it with "<start>".
            cur_vec = np.zeros((1,1))
            cur_vec[0,0] = start_vec
            cur_word = "<start>"
            output_sentence = ""
            # Start doing the feeding. Terminate when the model predicts an "<end>" or we reach the end
            # of the max target language sentence length.
            while cur_word != "<end>" and i < (self.len_target-1):
                i += 1
                if cur_word != "<start>":
                    output_sentence = output_sentence + "" + cur_word
                x_in = [cur_vec, sh, sc]
                # This will allow us to accomodate attention models, which we will talk about later.
                if attention:
                    x_in += [emb_out]
                [nvec, sh, sc] = infmodel.predict(x=x_in)
                # The output of the model is a massive softmax vector with one spot for every possible word. Convert
                # it to a word ID using argmax().
                cur_vec[0,0] = self.sample(nvec[0,0])
                cur_word = self.target_lang.idx2word[np.argmax(nvec[0,0])]
        return output_sentence
    
    
    def generate_batch_data_random(self, x, t , y, batch_size):
        ylen = len(y)
        loopcount = ylen // batch_size
        while (True):
            i = random.randint(0,loopcount)
            yield [x[i * batch_size:(i + 1) * batch_size], t[i * batch_size:(i + 1) * batch_size]], y[i * batch_size:(i + 1) * batch_size]
            
class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()
        
    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))
        self.vocab = sorted(self.vocab)
        self.word2idx["<pad>"] = 0
        self.idx2word[0] = "<pad>"
        for i,word in enumerate(self.vocab):
            self.word2idx[word] = i + 1
            self.idx2word[i+1] = word

    

# RNN "Cell" classes in Keras perform the actual data transformations at each timestep. Therefore, in order
# to add attention to LSTM, we need to make a custom subclass of LSTMCell.
class AttentionLSTMCell(LSTMCell):
    def __init__(self, **kwargs):
        self.attentionMode = False
        super(AttentionLSTMCell, self).__init__(**kwargs)
    
    # Build is called to initialize the variables that our cell will use. We will let other Keras
    # classes (e.g. "Dense") actually initialize these variables.
    @tf_utils.shape_type_conversion
    def build(self, input_shape):        
        # Converts the input sequence into a sequence which can be matched up to the internal
        # hidden state.
        self.dense_constant = TimeDistributed(Dense(self.units, name="AttLstmInternal_DenseConstant"))
        
        # Transforms the internal hidden state into something that can be used by the attention
        # mechanism.
        self.dense_state = Dense(self.units, name="AttLstmInternal_DenseState")
        
        # Transforms the combined hidden state and converted input sequence into a vector of
        # probabilities for attention.
        self.dense_transform = Dense(1, name="AttLstmInternal_DenseTransform")
        
        # We will augment the input into LSTMCell by concatenating the context vector. Modify
        # input_shape to reflect this.
        batch, input_dim = input_shape[0]
        batch, timesteps, context_size = input_shape[-1]
        lstm_input = (batch, input_dim + context_size)
        
        # The LSTMCell superclass expects no constant input, so strip that out.
        return super(AttentionLSTMCell, self).build(lstm_input)
    
    # This must be called before call(). The "input sequence" is the output from the 
    # encoder. This function will do some pre-processing on that sequence which will
    # then be used in subsequent calls.
    def setInputSequence(self, input_seq):
        self.input_seq = input_seq
        self.input_seq_shaped = self.dense_constant(input_seq)
        self.timesteps = tf.shape(self.input_seq)[-2]
    
    # This is a utility method to adjust the output of this cell. When attention mode is
    # turned on, the cell outputs attention probability vectors across the input sequence.
    def setAttentionMode(self, mode_on=False):
        self.attentionMode = mode_on
    
    # This method sets up the computational graph for the cell. It implements the actual logic
    # that the model follows.
    def call(self, inputs, states, constants):
        # Separate the state list into the two discrete state vectors.
        # ytm is the "memory state", stm is the "carry state".
        ytm, stm = states
        # We will use the "carry state" to guide the attention mechanism. Repeat it across all
        # input timesteps to perform some calculations on it.
        stm_repeated = K.repeat(self.dense_state(stm), self.timesteps)
        # Now apply our "dense_transform" operation on the sum of our transformed "carry state" 
        # and all encoder states. This will squash the resultant sum down to a vector of size
        # [batch,timesteps,1]
        # Note: Most sources I encounter use tanh for the activation here. I have found with this dataset
        # and this model, relu seems to perform better. It makes the attention mechanism far more crisp
        # and produces better translation performance, especially with respect to proper sentence termination.
        combined_stm_input = self.dense_transform(
            keras.activations.relu(stm_repeated + self.input_seq_shaped))
        # Performing a softmax generates a log probability for each encoder output to receive attention.
        score_vector = keras.activations.softmax(combined_stm_input, 1)
        # In this implementation, we grant "partial attention" to each encoder output based on 
        # it's log probability accumulated above. Other options would be to only give attention
        # to the highest probability encoder output or some similar set.
        context_vector = K.sum(score_vector * self.input_seq, 1)
        
        # Finally, mutate the input vector. It will now contain the traditional inputs (like the seq2seq
        # we trained above) in addition to the attention context vector we calculated earlier in this method.
        inputs = K.concatenate([inputs, context_vector])
        
        # Call into the super-class to invoke the LSTM math.
        res = super(AttentionLSTMCell, self).call(inputs=inputs, states=states)
        
        # This if statement switches the return value of this method if "attentionMode" is turned on.
        if(self.attentionMode):
            return (K.reshape(score_vector, (-1, self.timesteps)), res[1])
        else:
            return res

# Custom implementation of the Keras LSTM that adds an attention mechanism.
# This is implemented by taking an additional input (using the "constants" of the
# RNN class) into the LSTM: The encoder output vectors across the entire input sequence.
class LSTMWithAttention(RNN):
    def __init__(self, units, **kwargs):
        cell = AttentionLSTMCell(units=units)
        self.units = units
        super(LSTMWithAttention, self).__init__(cell, **kwargs)
        
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.input_dim = input_shape[0][-1]
        self.timesteps = input_shape[0][-2]
        return super(LSTMWithAttention, self).build(input_shape) 
    
    # This call is invoked with the entire time sequence. The RNN sub-class is responsible
    # for breaking this up into calls into the cell for each step.
    # The "constants" variable is the key to our implementation. It was specifically added
    # to Keras to accomodate the "attention" mechanism we are implementing.
    def call(self, x, constants, **kwargs):
        if isinstance(x, list):
            self.x_initial = x[0]
        else:
            self.x_initial = x
        
        # The only difference in the LSTM computational graph really comes from the custom
        # LSTM Cell that we utilize.
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        self.cell.setInputSequence(constants[0])
        return super(LSTMWithAttention, self).call(inputs=x, constants=constants, **kwargs)

    # Re-create an entirely new model and set of layers for the attention model
    

        
test = seq2seq_Attention_all('P3_big_BN.h5', 'QA_all_char.txt')
test.train()
model = test.create_model()
encoder_Inference, decoder_Inference = test.createAttentionInference(model)
while True:
    text = input('【input Answer】 \n' )
    result = test.translate(text, encoder_Inference, decoder_Inference, True)
    print('【output question】 \n')
    print(result)
#print(test.translate('台幣存款利率查詢查詢方式說明如下：◎ 台幣利率請參閱官網[link url="https://www.megabank.com.tw/other/bulletin02_05.asp"]新臺幣存放款利率表[/link]。', encoder_Inference, decoder_Inference, True))
#
#with open(test.path_to_file, 'r', encoding='utf8') as rfile:
#    data = rfile.readlines()
#
#with open('generator.csv', 'w',encoding='utf-8-sig') as wfile:
#    writer = csv.writer(wfile)
#    
#    for d in data[:100]:
#        A, Q = d.split('\t')
#        
#        writer.writerow([A, test.translate(A, encoder_Inference, decoder_Inference, True)])
#
#while(True):
#    test_text = input('【input Answer】 \n' )
#    result = test.translate(str(test_text), encoder_Inference, decoder_Inference, True)
#    print('【output question】 \n', result)
                     

                     
                     