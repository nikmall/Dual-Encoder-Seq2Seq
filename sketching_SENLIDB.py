# IMPORTING THE APPROPRIATE LIBRARIES
import pandas as pd
import numpy as np
import sql_metadata as smeta
from sql_metadata import generalize_sql

# =============================================================================
# TRAIN SKETCH PRODUCTION. In the SENLIDB dataset we have the sql query and the 
# natural language discription explicitly so we don not need to produce the sql query
# from skratch
# =============================================================================

# we load the datasets
data_train = pd.read_json('SENLIDB/train.json')
sql_train = data_train['sql']

# in this part we have all the sql reserved words that sql uses in order not to replace them during the 
# sql sketch phase
reserved = pd.read_csv('SENLIDB/reserved.csv')
reserved = reserved.values.tolist()
reserved = sum(reserved, [])
reserved = set(reserved)

#in this loop we replace the table names, the columns and the aliases in order the schema to be agnostic 
for query in range(sql_train.shape[0]):
    
    # we employ the sql metadata library for the sketch production. This lib grabs from a query the table names, 
    # the aliases and the column names
    sql_train.loc[query] = generalize_sql(sql_train.loc[query])
    temp = sql_train.loc[query].lower().replace('as', '')
    columns = smeta.get_query_columns(temp)
    tables = smeta.get_query_tables(sql_train.loc[query])
    alias = smeta.get_query_table_aliases(sql_train.loc[query])
    
    # in this piece of code we make the schema agnostic according to the previous preprocessing (table-wise)
    for elem in range(len(tables)):
        if tables[elem].upper() not in reserved:
            sql_train.loc[query] = sql_train.loc[query].lower().replace(tables[elem].lower() , 'table')

    # in this piece of code we make the schema agnostic according to the previous preprocessing (column-wise)
    for elem in range(len(columns)):
        if columns[elem].upper() not in reserved:
            sql_train.loc[query] = sql_train.loc[query].lower().replace(columns[elem].lower() , 'alias.col')

    # in this piece of code we make the schema agnostic according to the previous preprocessing (alias-wise)   
    for elem in alias:
        sql_train.loc[query] = sql_train.loc[query].lower().replace(''.join(list(alias.keys())) , 'alias')
        
    print(f'{query} out of {sql_train.shape[0]} done!')
    
# we save the sketch in a .csv format    
sql_train.to_csv('sketch_train.csv' , index = False)

# we load the appropriate datasets again
sql_sketch = pd.read_csv('SENLIDB/sketch_train.csv')
data_train['sql'] = sql_sketch
data_train = data_train[data_train['description'].str.count('') > 1]
data_train = data_train[['description' , 'sql']]

data_train['description'].to_csv('description_sketch_train.csv' , index = False) 
data_train['sql'].to_csv('sql_sketch_train.csv' , index = False)
    
# =============================================================================
# TEST SKETCH PRODUCTION. In this part we do exacty the same things, (more or less the same code lines)
# in order to make the test part of the dataset as schema agnostic via the production of the sketches.
# =============================================================================

data_test = pd.read_json('SENLIDB/test.json')
data_test = data_test.drop(columns = ['description' , 'title' , 'url' , 'comments' , 'sql_plain' , 'id'])
sql_test = data_test['sql']


reserved = pd.read_csv('SENLIDB/reserved.csv')
reserved = reserved.values.tolist()
reserved = sum(reserved, [])
reserved = set(reserved)


for query in range(sql_test.shape[0]):
    
    sql_test.loc[query] = generalize_sql(sql_test.loc[query])
    temp = sql_test.loc[query].lower().replace('as', '')
    columns = smeta.get_query_columns(temp)
    tables = smeta.get_query_tables(sql_test.loc[query])
    alias = smeta.get_query_table_aliases(sql_test.loc[query])

    for elem in range(len(tables)):
        if tables[elem].upper() not in reserved:
            sql_test.loc[query] = sql_test.loc[query].lower().replace(tables[elem].lower() , 'table')

    for elem in range(len(columns)):
        if columns[elem].upper() not in reserved:
            sql_test.loc[query] = sql_test.loc[query].lower().replace(columns[elem].lower() , 'alias.col')
    
    for elem in alias:
        sql_test.loc[query] = sql_test.loc[query].lower().replace(''.join(list(alias.keys())) , 'alias')
        
    print(f'{query} out of {sql_test.shape[0]} done!')
    
    
sql_test.to_csv('sketch_test.csv' , index = False)


# pd.json_normalize(data_test['annotations'][0])

# =============================================================================
# OPEN NMNT TRAIN AND PREPROCESS              
# =============================================================================
 
#THESE COMMANDS ARE FOR TRAINING THE SEQ2SEQ MODEL USING THE OPEN NMT LIBRARY 
# WE RUN THE CODE (lines 115 - 118) IN A COMMAND LINE ENVIRONMENT AND WE GET BACK A .CSV FILE WITH THE CORRESPONDING PREDICTIONS            


# SENLIDB DATASET
# onmt_preprocess -train_src data/description_sketch_train.csv -train_tgt data/sql_sketch_train.csv -valid_src data/description_sketch_train.csv -valid_tgt data/sql_sketch_train.csv -save_data data/data
# python .\tools\embeddings_to_torch.py  -emb_file_both "C:\Users\Giorgos\glove_dir\glove.6B.100d.txt" -output_file "data/embeddings" -dict_file "data/data.vocab.pt"
# onmt_train -save_model data/model -batch_size 64 -layers 2 -rnn_size 500 -learning_rate 1 -learning_rate_decay 0.5 -report_every 1 -train_steps 25 -word_vec_size 100 -pre_word_vecs_enc "data/embeddings.enc.pt" -pre_word_vecs_dec "data/embeddings.dec.pt" -data data/data
# onmt_translate -model data/model_step_25.pt -src data/description_sketch_train.csv -output preds_SENLIDB.csv -replace_unk -verbose
