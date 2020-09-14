#importing the appropriate libraries
import pandas as pd
import numpy as np
import sql_metadata as smeta
from sql_metadata import generalize_sql
import re

# =============================================================================
# train sql production. In this part we are to produce the sql query.
# the dataset does not provide the sql query explicitly but in table form.
# we need to combine these tables for the final sql statement
# =============================================================================

#importing the appropriate json files
data_train = pd.read_json('WikiSQL/train.jsonl' , lines = True)
data_train_tables = pd.read_json('WikiSQL/train.tables.jsonl' , lines = True)

data_train = data_train.drop(['phase'] , axis = 1)
data_train_tables = data_train_tables[['id' , 'header']]
agg_ops = list(enumerate(['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']))
cond_ops = list(enumerate(['=', '>', '<', 'OP']))

# we rename the columns appopriate for the final joins
data_train = data_train.rename(columns = {'table_id' : 'id'})    

#join of of the dataframes
sql_table = pd.merge(data_train , data_train_tables , on = 'id')

#we enumerate the dataset for better handling accoring to the frame type
for elem in range(sql_table.shape[0]):
    sql_table['header'][elem] = list(enumerate(sql_table['header'][elem]))

# in order to handle the column names as an one entity (one word) we replace the ' ' character with '_'
for elem1 in range(sql_table.shape[0]):
    for elem2 in range(len(sql_table['header'][elem1])):
        sql_table['header'][ elem1 ][elem2] = sql_table['header'][ elem1 ][ elem2 ][1].replace(' ' , '_')
        sql_table['header'][ elem1 ][elem2] = (elem2 , sql_table['header'][ elem1 ][elem2])
    
    if len(sql_table['sql'][elem1]['conds']) != 0:
        if type(sql_table['sql'][elem1]['conds'][0][2]) is str:
            sql_table['sql'][elem1]['conds'][0][2] = sql_table['sql'][elem1]['conds'][0][2].replace(' ' , '_')
    print(elem1)    
    
# in this part we construct the sql statement after the aformentioned preprocess according to all cases   
sql_sketch = []            
        
for elem in range(sql_table.shape[0]):
    
    if len(list(sql_table['sql'][ elem ].values())[1]) == 3:
        temp1 = list(sql_table['sql'][ elem ].values())[0]
        temp2 = list(sql_table['sql'][ elem ].values())[1][0][0]
        temp3 = list(sql_table['sql'][ elem ].values())[1][0][1]
        temp = 'select ' + str(sql_table['header'][ elem ][temp1][1]) + ' from table where ' + str(sql_table['header'][ elem ][ temp2 ][1]) + ' ' + (cond_ops[temp3][1]) + ' ' + str(list(sql_table['sql'][ elem ].values())[1][0][2])
        sql_sketch.append(temp)
        
    elif len(list(sql_table['sql'][ elem ].values())[1]) == 0:
        temp1 = list(sql_table['sql'][ elem ].values())[0]
        temp2 = list(sql_table['sql'][ elem ].values())[2]
        temp = 'select ' +  (agg_ops[temp2][1]) + ' ' + sql_table['header'][ elem ][temp1][1] + ' from table'
        sql_sketch.append(temp)
        
    else:
        temp1 = list(sql_table['sql'][ elem ].values())[0]
        temp2 = list(sql_table['sql'][ elem ].values())[2]
        temp3 = list(sql_table['sql'][ elem ].values())[1][0][1]
        temp4 = list(sql_table['sql'][ elem ].values())[1][0][0]
        temp = 'select ' +  (agg_ops[temp2][1]) + ' ' + sql_table['header'][ elem ][temp1][1] + ' from table where ' + str(sql_table['header'][ elem ][ temp4 ][1]) + ' ' + (cond_ops[temp3][1]) + ' ' + str(list(sql_table['sql'][ elem ].values())[1][0][2])
        sql_sketch.append(temp)
     
    
    print(f'{elem} out of {sql_table.shape[0]} done!')        
    
#we save the datasets as .csv format
sql_table['sql_sketch'] = sql_sketch
sql_table['sql_sketch'].to_csv('sql_train_WIki.csv' , index = False)

# =============================================================================
# sketch sql production train
# =============================================================================

# we import the .csv file we produced above
sql_train = pd.read_csv('WikiSQL/sql_train_Wiki.csv')

#we keep ALL the reserved words used by sql language in order not to be replaced during the sketching production process
reserved = pd.read_csv('WikiSQL/reserved.csv')
reserved = reserved.values.tolist()
reserved = sum(reserved, [])
reserved = set(reserved)

# we keep all the aggragate functions in a list for better and faster exclusion handlings
agg = ['TOP', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']

#we produce the sql sketch according to all possible cases
for query in range(sql_train.shape[0]):
    sql_train.loc[query] = sql_train.loc[query].str.split(' ')
    
    if sql_train.loc[query][0][2].upper() not in agg and len(sql_train.loc[query][0]) != 5 and len(sql_train.loc[query][0]) != 8:
        
        sql_train.loc[query][0] = list(map(lambda st: str.replace(st, sql_train.loc[query][0][2], "column"), sql_train.loc[query][0])) 
        sql_train.loc[query][0] = list(map(lambda st: str.replace(st, sql_train.loc[query][0][6], "column"), sql_train.loc[query][0])) 
        sql_train.loc[query][0] = list(map(lambda st: str.replace(st, sql_train.loc[query][0][8], "ct"), sql_train.loc[query][0])) 
        sql_train.loc[query][0] = ' '.join(sql_train.loc[query][0])
        
    elif sql_train.loc[query][0][2].upper() not in agg and len(sql_train.loc[query][0]) == 8:
        
        sql_train.loc[query][0] = list(map(lambda st: str.replace(st, sql_train.loc[query][0][1], "column"), sql_train.loc[query][0])) 
        sql_train.loc[query][0] = list(map(lambda st: str.replace(st, sql_train.loc[query][0][5], "column"), sql_train.loc[query][0])) 
        sql_train.loc[query][0] = list(map(lambda st: str.replace(st, sql_train.loc[query][0][7], "ct"), sql_train.loc[query][0])) 
        sql_train.loc[query][0] = ' '.join(sql_train.loc[query][0])
     
    elif sql_train.loc[query][0][1] == '' and len(sql_train.loc[query][0]) == 9 and sql_train.loc[query][0][2].upper() in agg:
       
        sql_train.loc[query][0] = list(map(lambda st: str.replace(st, sql_train.loc[query][0][6], "column"), sql_train.loc[query][0])) 
        sql_train.loc[query][0] = list(map(lambda st: str.replace(st, sql_train.loc[query][0][8], "ct"), sql_train.loc[query][0])) 
        sql_train.loc[query][0] = ' '.join(sql_train.loc[query][0])

    elif len(sql_train.loc[query][0]) == 5:
        
        sql_train.loc[query][0] = list(map(lambda st: str.replace(st, sql_train.loc[query][0][2], "column"), sql_train.loc[query][0]))
        sql_train.loc[query][0] = ' '.join(sql_train.loc[query][0])
    
    else:
        
        sql_train.loc[query][0] = list(map(lambda st: str.replace(st, sql_train.loc[query][0][3], "column"), sql_train.loc[query][0])) 
        sql_train.loc[query][0] = list(map(lambda st: str.replace(st, sql_train.loc[query][0][7], "column"), sql_train.loc[query][0])) 
        sql_train.loc[query][0] = list(map(lambda st: str.replace(st, sql_train.loc[query][0][9], "ct"), sql_train.loc[query][0])) 
        sql_train.loc[query][0] = ' '.join(sql_train.loc[query][0])
    
    print(query)
    
#we save the sketches and the corresponding natural language question in a .csv file   
sql_train.to_csv('sql_sketch_train_Wiki.csv' , index = False)
data_train['question'].to_csv('sql_description_train_Wiki.csv' , index = False)

# =============================================================================
# test sql production
# =============================================================================

# =============================================================================
# IN THIS PART WE FOLLOW EXACTLY THE SAME STEPS AS BEFORE BUT FOR THE TEST SET
# =============================================================================
data_test = pd.read_json('WikiSQL/test.jsonl' , lines = True)
data_test_tables = pd.read_json('WikiSQL/test.tables.jsonl' , lines = True)

data_test = data_test.drop(['phase'] , axis = 1)
data_test_tables = data_test_tables[['id' , 'header']]
agg_ops = list(enumerate(['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']))
cond_ops = list(enumerate(['=', '>', '<', 'OP']))

data_test = data_test.rename(columns = {'table_id' : 'id'})    

sql_table = pd.merge(data_test , data_test_tables , on = 'id')

for elem in range(sql_table.shape[0]):
    sql_table['header'][elem] = list(enumerate(sql_table['header'][elem]))

for elem1 in range(sql_table.shape[0]):
    for elem2 in range(len(sql_table['header'][elem1])):
        sql_table['header'][ elem1 ][elem2] = sql_table['header'][ elem1 ][ elem2 ][1].replace(' ' , '_')
        sql_table['header'][ elem1 ][elem2] = (elem2 , sql_table['header'][ elem1 ][elem2])
    
    if len(sql_table['sql'][elem1]['conds']) != 0:
        if type(sql_table['sql'][elem1]['conds'][0][2]) is str:
            sql_table['sql'][elem1]['conds'][0][2] = sql_table['sql'][elem1]['conds'][0][2].replace(' ' , '_')
    print(elem1)    
    
     
sql_sketch = []            
        
for elem in range(sql_table.shape[0]):
    
    if len(list(sql_table['sql'][ elem ].values())[1]) == 3:
        temp1 = list(sql_table['sql'][ elem ].values())[0]
        temp2 = list(sql_table['sql'][ elem ].values())[1][0][0]
        temp3 = list(sql_table['sql'][ elem ].values())[1][0][1]
        temp = 'select ' + str(sql_table['header'][ elem ][temp1][1]) + ' from table where ' + str(sql_table['header'][ elem ][ temp2 ][1]) + ' ' + (cond_ops[temp3][1]) + ' ' + str(list(sql_table['sql'][ elem ].values())[1][0][2])
        sql_sketch.append(temp)
        
    elif len(list(sql_table['sql'][ elem ].values())[1]) == 0:
        temp1 = list(sql_table['sql'][ elem ].values())[0]
        temp2 = list(sql_table['sql'][ elem ].values())[2]
        temp = 'select ' +  (agg_ops[temp2][1]) + ' ' + sql_table['header'][ elem ][temp1][1] + ' from table'
        sql_sketch.append(temp)
        
    else:
        temp1 = list(sql_table['sql'][ elem ].values())[0]
        temp2 = list(sql_table['sql'][ elem ].values())[2]
        temp3 = list(sql_table['sql'][ elem ].values())[1][0][1]
        temp4 = list(sql_table['sql'][ elem ].values())[1][0][0]
        temp = 'select ' +  (agg_ops[temp2][1]) + ' ' + sql_table['header'][ elem ][temp1][1] + ' from table where ' + str(sql_table['header'][ elem ][ temp4 ][1]) + ' ' + (cond_ops[temp3][1]) + ' ' + str(list(sql_table['sql'][ elem ].values())[1][0][2])
        sql_sketch.append(temp)
     
    
    print(f'{elem} out of {sql_table.shape[0]} done!')        
             
sql_table['sql_sketch'] = sql_sketch

sql_table['sql_sketch'].to_csv('sql_test_WIki.csv' , index = False)

# =============================================================================
# sketch sql production test
# =============================================================================

sql_test = pd.read_csv('WikiSQL/sql_test_Wiki.csv')
reserved = pd.read_csv('WikiSQL/reserved.csv')
reserved = reserved.values.tolist()
reserved = sum(reserved, [])
reserved = set(reserved)

agg = ['TOP', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
for query in range(sql_test.shape[0]):
    
    sql_test.loc[query] = sql_test.loc[query].str.split(' ')
    
    if sql_test.loc[query][0][2].upper() not in agg and len(sql_test.loc[query][0]) != 5 and len(sql_test.loc[query][0]) != 8:
        
        sql_test.loc[query][0] = list(map(lambda st: str.replace(st, sql_test.loc[query][0][2], "column"), sql_test.loc[query][0])) 
        sql_test.loc[query][0] = list(map(lambda st: str.replace(st, sql_test.loc[query][0][6], "column"), sql_test.loc[query][0])) 
        sql_test.loc[query][0] = list(map(lambda st: str.replace(st, sql_test.loc[query][0][8], "ct"), sql_test.loc[query][0])) 
        sql_test.loc[query][0] = ' '.join(sql_test.loc[query][0])
        
    elif sql_test.loc[query][0][2].upper() not in agg and len(sql_test.loc[query][0]) == 8:
        
        sql_test.loc[query][0] = list(map(lambda st: str.replace(st, sql_test.loc[query][0][1], "column"), sql_test.loc[query][0])) 
        sql_test.loc[query][0] = list(map(lambda st: str.replace(st, sql_test.loc[query][0][5], "column"), sql_test.loc[query][0])) 
        sql_test.loc[query][0] = list(map(lambda st: str.replace(st, sql_test.loc[query][0][7], "ct"), sql_test.loc[query][0])) 
        sql_test.loc[query][0] = ' '.join(sql_test.loc[query][0])
     
    elif sql_test.loc[query][0][1] == '' and len(sql_test.loc[query][0]) == 9 and sql_test.loc[query][0][2].upper() in agg:
       
        sql_test.loc[query][0] = list(map(lambda st: str.replace(st, sql_test.loc[query][0][6], "column"), sql_test.loc[query][0])) 
        sql_test.loc[query][0] = list(map(lambda st: str.replace(st, sql_test.loc[query][0][8], "ct"), sql_test.loc[query][0])) 
        sql_test.loc[query][0] = ' '.join(sql_test.loc[query][0])

    elif len(sql_test.loc[query][0]) == 5:
        
        sql_test.loc[query][0] = list(map(lambda st: str.replace(st, sql_test.loc[query][0][2], "column"), sql_test.loc[query][0]))
        sql_test.loc[query][0] = ' '.join(sql_test.loc[query][0])
    

    else:
        
        sql_test.loc[query][0] = list(map(lambda st: str.replace(st, sql_test.loc[query][0][3], "column"), sql_test.loc[query][0])) 
        sql_test.loc[query][0] = list(map(lambda st: str.replace(st, sql_test.loc[query][0][7], "column"), sql_test.loc[query][0])) 
        sql_test.loc[query][0] = list(map(lambda st: str.replace(st, sql_test.loc[query][0][9], "ct"), sql_test.loc[query][0])) 
        sql_test.loc[query][0] = ' '.join(sql_test.loc[query][0])
    
    print(query)
    
    
sql_test.to_csv('sql_sketch_test_Wiki.csv' , index = False)
data_test['question'].to_csv('sql_description_test_Wiki.csv' , index = False)
        
            
# =============================================================================
# OPEN NMNT TRAIN AND PREPROCESS              
# =============================================================================
 
#THESE COMMANDS ARE FOR TRAINING THE SEQ2SEQ MODEL USING THE OPEN NMT LIBRARY 
# WE RUN THE CODE (lines 261 - 264) IN A COMMAND LINE ENVIRONMENT AND WE GET BACK A .CSV FILE WITH THE CORRESPONDING PREDICTIONS            
             
# WIKISQL DATASET
# onmt_preprocess -train_src data/sql_description_train_Wiki.csv -train_tgt data/sql_sketch_train_Wiki.csv -valid_src data/sql_description_train_Wiki.csv -valid_tgt data/sql_sketch_train_Wiki.csv -save_data data/data_wiki
# python .\tools\embeddings_to_torch.py  -emb_file_both "C:\Users\Giorgos\glove_dir\glove.6B.100d.txt" -output_file "data/embeddings" -dict_file "data/data_wiki.vocab.pt"
# onmt_train -save_model data/model -batch_size 64 -layers 2 -rnn_size 500 -learning_rate_decay 0.5 -learning_rate 1 -report_every 1 -train_steps 25 -word_vec_size 100 -pre_word_vecs_enc "data/embeddings.enc.pt" -pre_word_vecs_dec "data/embeddings.dec.pt" -data data/data_wiki
# onmt_translate -model data/model_step_25.pt -src data/sql_description_train_Wiki.csv -output preds_Wiki.csv -replace_unk -verbose
