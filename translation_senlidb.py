# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd 
import os,unicodedata, re, io, time, gc, warnings
import tensorflow as tf

warnings.filterwarnings("ignore")


""" Pre-processing  preparing natural  language for training
        Removes unwated symbols and spaces. Also, adds <start>  <end> """
def preprocess_sentence(w):
    
  w = w.lower().strip()
  # creating a space between a word and the punctuation following it  eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
  w = re.sub(r"([?.!,¿();])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
  w = w.strip()

  # add start and an end token to the sentence to know when to start and stop predicting.
  w = '<start> ' + w + ' <end>'
  return w

""" Pre-processing  preparing sql sentences  for training
    Removes unwated symbols and spaces. Also, adds <start>  <end>"""
def preprocess_sql(w):
    
  w = w.lower().strip()
  w = re.sub(r"([?.!,¿();])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  w = re.sub(r"[^a-zA-Z.()]+", " ", w)
  w = w.strip()
  
  # add start and an end token to the sentence to know when to start and stop predicting.
  w = '<start> ' + w + ' <end>'
  return w


""" Perform tokenization on the data along with padding on the sentences max length"""
def tokenize(x_train,x_test):
    
    language = np.concatenate((x_train,x_test), axis=0)
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(language)
    
    tensor_train = tokenizer.texts_to_sequences(x_train)
    tensor_train = tf.keras.preprocessing.sequence.pad_sequences(tensor_train,padding='post')
    print('tensor_train.shape', tensor_train.shape)

    tensor_test = tokenizer.texts_to_sequences(x_test)
    tensor_test = tf.keras.preprocessing.sequence.pad_sequences(
        tensor_test, maxlen= tensor_train.shape[1], padding='post')    
    print('tensor_test.shape', tensor_test.shape)
    
    return tensor_train, tensor_test, tokenizer


"""Method that performs the nldib dataset creation. 
   Removes empty lines, apply preprocessing and tokenization"""
def create_nldib_dataset(filepath_train, filepath_test):
    
    train_set = pd.read_json(filepath_train)
    test_set = pd.read_json(filepath_test)
    
    #replace empty strings with nan for better management
    train_set.replace('', np.nan, inplace=True)
    test_set.replace('', np.nan, inplace=True)
    
    # drop empty rows
    train_clean = train_set.dropna(subset=['description', 'sql']) #
    test_clean = test_set.dropna(subset=['description', 'sql'])
    
    # preproces sentences (both description and sql)
    train_clean = train_clean[['description', 'sql']]
    test_clean = test_clean[['description', 'sql']]
    
    #test_clean = test_clean.apply(str).applymap(preprocess_sentence)
    x_train, y_train = train_clean['description'], train_clean['sql']
    x_train = x_train.apply(str).apply(preprocess_sentence).values
    y_train = y_train.apply(str).apply(preprocess_sql).values
    
    x_test, y_test = test_clean['description'], test_clean['sql']
    # apply pre-processiing and convert to numpy array
    x_test = x_test.apply(str).apply(preprocess_sentence).values
    y_test = y_test.apply(str).apply(preprocess_sql).values
    
    input_tensor_train, input_tensor_test, inp_lang_tokenizer = tokenize(x_train,x_test)
    target_tensor_train, target_tensor_test, targ_lang_tokenizer = tokenize(y_train,y_test)
    
    return input_tensor_train, input_tensor_test, target_tensor_train, target_tensor_test, inp_lang_tokenizer, targ_lang_tokenizer

"""Method that performs the sketches of nldib dataset preparation. 
   Removes empty lines, apply preprocessing and tokenization"""
def create_sketch_data(file_sketch_train, file_sketch_test, tokenizer):
    
    x_train = pd.read_csv(file_sketch_train)
    x_test = pd.read_csv(file_sketch_test)
    
    x_train = x_train.sql.apply(str).apply(preprocess_sentence).values
    x_test = x_test.sql.apply(str).apply(preprocess_sentence).values
    
    language = np.concatenate((x_train,x_test), axis=0)
    
    #fit/update on existing tokenizer
    tokenizer.fit_on_texts(language)
    
    tensor_train = tokenizer.texts_to_sequences(x_train)
    tensor_train = tf.keras.preprocessing.sequence.pad_sequences(tensor_train,padding='post')
    print('tensor_train_sketch.shape', tensor_train.shape)

    tensor_test = tokenizer.texts_to_sequences(x_test)
    tensor_test = tf.keras.preprocessing.sequence.pad_sequences(
        tensor_test, maxlen= tensor_train.shape[1], padding='post')    
    print('tensor_test_sketch.shape', tensor_test.shape)
    
    return tensor_train, tensor_test


"""Method that that returns the target test data for evaluation"""
def get_sql_test(filepath_test):
    
    test_set = pd.read_json(filepath_test)
    test_set.replace('', np.nan, inplace=True)
    test_clean = test_set.dropna(subset=['description', 'sql'])
    
    test_clean = test_clean[['description', 'sql']]
    
    y_test =test_clean['sql']

    return y_test
    


""" Encoder class """
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))
    


# Attention Class based on the tensorflow guide implementation
class Attention(tf.keras.layers.Layer):
    
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):

        #  broadcast  along the time axis 
        query_with_time_axis = tf.expand_dims(query, 1)

        # calculate the attention score 
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        # calculate weights of attention
        attention_weights = tf.nn.softmax(score, axis=1)
        # calculate context
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
    

# Decoder class with dual attention
class Decoder(tf.keras.Model):
    
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # Decoder dual attentions
        self.attention1 = Attention(self.dec_units)
        self.attention2 = Attention(self.dec_units)


    def call(self, x, hidden1, enc_output1, hidden2, enc_output2):
        
        #Calculate two attentions for the two encoders inputs
        context_vector1, attention_weights1 = self.attention1(hidden1, enc_output1)
        context_vector2, attention_weights2 = self.attention2(hidden2, enc_output2)

        x = self.embedding(x)

        # concatenate the context along with the x input
        x = tf.concat([tf.expand_dims(context_vector1, 1), x], axis=-1)
        x = tf.concat([tf.expand_dims(context_vector2, 1), x], axis=-1)
        
        # passing the concatenated vector to the rnn
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights1, attention_weights2
 
""" START """ 

# file paths
filepath_train = 'SENLIDB/train.json'
filepath_test = 'SENLIDB/test.json'
file_sketch_train = 'SENLIDB/sql_sketch_train.csv'
file_sketch_test = 'SENLIDB/sketch_test.csv'

#tensors of data of normal data
input_tensor_train, input_tensor_test, target_tensor_train, target_tensor_test, inp_lang, targ_lang  = create_nldib_dataset(filepath_train, filepath_test)
max_length_targ, max_length_inp = target_tensor_train.shape[1], input_tensor_train.shape[1]

#tensors of data of sketch data
input_tensor_train_sketch, input_tensor_test_sketch  = create_sketch_data(
    file_sketch_train, file_sketch_test, inp_lang)
max_length_inp_sketch = input_tensor_train_sketch.shape[1]


# parameters of model training
#BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 30  # 500 paper
units = 30 # 500 paper
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
EPOCHS = 25


# create tensorflow dataset 

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, 
                                              input_tensor_train_sketch,
                                              target_tensor_train))#.shuffle(BUFFER_SIZE)

dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)

    
 
#Create encoders 
encoder1 = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
encoder2 = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

# Create Optimizer and Loss function
optimizer = tf.keras.optimizers.Adamax()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# create loss function
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


""" Training step function. The whole model process is here.
    We create two encoders with default initialization. 
    Then we pass their outputs to the decoder along with the training 
    sentence  target. This is done step by step producing the next output
    for the target required output. We calculate the loss and finally
    the gradient of the variables"""
@tf.function
def train_step(inp, inp_sketch, targ, enc_hidden1, enc_hidden2):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output1, enc_hidden1 = encoder1(inp, enc_hidden1)
        
        enc_output2, enc_hidden2 = encoder2(inp_sketch, enc_hidden2)

        dec_hidden1 = enc_hidden1
        dec_hidden2 = enc_hidden2

        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_outputs  to the decoder
            predictions, dec_hidden, _, _ = decoder(dec_input, 
                                                    dec_hidden1, enc_output1,
                                                    dec_hidden2, enc_output2
                                                    )
            loss += loss_function(targ[:, t], predictions)
            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder1.trainable_variables + encoder2.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

""" Here the training is part run for epochs"""

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden1 = encoder1.initialize_hidden_state()
    enc_hidden2 = encoder2.initialize_hidden_state()

    total_loss = 0

    for (batch, (inp, inp_sketch, targ)) in enumerate(dataset.take(steps_per_epoch)):
        
        batch_loss = train_step(inp, inp_sketch, targ, enc_hidden1, enc_hidden2)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    

""" predict function of given input sentence and sketch.
    Passes the inputs to the encoders and then to the decoder along 
    with the <start> string and the decoder generates the rest of the 
    string. It stops when the model predicts <end> or when we reach the
    max string length"""

def predict(sentence, sketch):

  in_sentence = sentence.reshape(1,-1)
  in_sketch = sketch.reshape(1,-1)

  result = ''

  hidden = [tf.zeros((1, units))]
  enc_output1, enc_hidden1 = encoder1(in_sentence, hidden)
  enc_output2, enc_hidden2 = encoder2(in_sketch, hidden)
  
  dec_hidden1 = enc_hidden1
  dec_hidden2 = enc_hidden2

  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

  for t in range(max_length_targ):
    predictions, dec_hidden, attention_weights1, attention_weights2 = \
        decoder(dec_input, dec_hidden1, enc_output1,dec_hidden2, enc_output2)
        
    predicted_id = tf.argmax(predictions[0]).numpy()

    result += targ_lang.index_word[predicted_id] + ' '

    if targ_lang.index_word[predicted_id] == '<end>':
      return result

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result

    
''' Prediction stage of test data
    We pass the results of each test x input to the evaluate and finally
    the results of the x input along with the result to find the blue score
 ''' 

y_test = get_sql_test(filepath_test)

predictions = []

    
for xtest in zip(input_tensor_test, input_tensor_test_sketch):
    predictions.append(predict(xtest[0], xtest[1])) 
    
    
from  nltk.translate import bleu_score
print('BLEU score: {}'.format(bleu_score.corpus_bleu(predictions, y_test)))

