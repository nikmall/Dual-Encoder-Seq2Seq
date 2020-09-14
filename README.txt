1)SKETCH AND SQL PRODUCTION

SENLIDB dataset
1. Open and load the 'sketching_SENLIDB.py' file.
2. After after putting the 'train.json' and 'test.json' files into the correct directory that Python pings, run the code.
3. The program creates  the 'description_sketch_train.csv' that contains the natural language of each query, the 'sql_sketch_train.csv' that contains the 
schema agnostic SQL sketch and the 'sketch_test.csv' that keeps the schema agnostic SQL for the test part.
4. At the end of the file, in commented lines 115-118, there is the code for the OpenNMT-py training. After installing the approriate pacakages (OpenNMT and Glove stated
in the links of the report) run the code in order to train the embeddings and feed the first SEQ2SEQ model.

WikiSQL dataset
1. Open and load the 'sketching_WikiSQL.py' file.
2. After after putting the 'train.jsonl', 'test.jsonl', 'train.tables.jsonl' and 'test.tables.jsonl' files into the correct directory that Python pings, run the code.
3. The program creates the 
	'sql_train_WIki.csv' that contains the true SQL statements that the train dataset provides (the dataset does not give the SQL queries explicitly 
but they need construction), 
	the 'sql_sketch_train_Wiki.csv' file that is the SQL train sketch schema agnostic statements (no columns, tables or aliases), 
	the 'sql_description_train_Wiki.csv' that contains the corresponding natural language queries for the training set, 
	the 'sql_test_WIki.csv' that contains the true SQL statements that the test dataset provides, 
	the 'sql_sketch_test_Wiki.csv' file that is the SQL test sketch schema agnostic statements and 
	the 'sql_description_test_Wiki.csv' that contains the corresponding natural language queries for the testing set
4. At the end of the file, in commented lines 263 - 266, there is the code for the OpenNMT-py training. After installing the approriate pacakages (OpenNMT and Glove stated
in the links of the report) run the code in order to train the embeddings and feed the first SEQ2SEQ model.

You will also get a 'reserved.csv' file that contains all possible SQL commands. This is very useful in our code during the phase of sketching production not to 
change a word that is reserved accidentally.

2) Dual Encoder Seq2Seq Model
In order to run the Dual Encoder  Model, tensorflow 2.1 is required to be installed along with nltk for blue score calculation alogn with  pandas and numpy.
They are two python programs one for each dataset. The files are translation_senlidbpy and translation_wiki.py. Executing them performs 
all the tasks of pre-processing model train and prediction with finally printing the blue score. 
The train and test datasets are required to be found in the folders local SENLIDB and WikiSql. 
The data files  are  included in the project repository. 