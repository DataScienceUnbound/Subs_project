import boto3
import json
import pandas as pd 
import io
#import pickle as p
import dill as p

def s3_list(prefix=''):
    files = []
    resource = boto3.resource('s3') 
    data_bucket = resource.Bucket('datascience.unbound.com')
    for obj in data_bucket.objects.filter(Prefix =prefix):
        files.append(obj.key)
    return files

def get_last_books_csv_fname(folder='books_csv_files'):
    timestamps = []
    for book_fname in s3_list(folder):
        # e.g. 'books_csv_files/books-2018-04-27 01_00_56.csv'

        try:
            time_stamp = '20'+book_fname.split('-20')[1].split('.')[0]
            ts = time_stamp.replace('_',':')
            timestamps.append((pd.to_datetime(ts,yearfirst=True),book_fname))
        except:
            pass
    # time stamps is a list of tuples
    return max(timestamps)[1]

def get_last_file_with_prefix(prefix):
    ''' Only works if no other date that books csv suffix  '''
    timestamps = []
    for fname in s3_list(prefix):
        # e.g. 'books_csv_files/books-2018-04-27 01_00_56.csv'
        try:
            time_stamp = '20'+fname.split('-20')[1].split('.')[0]
            ts = time_stamp.replace('_',':')
            timestamps.append((pd.to_datetime(ts,yearfirst=True),fname))
        except:
             pass
            # time stamps is a list of tuples
    return max(timestamps)[1]

def load_last_books_csv(folder='books_csv_files'):
	key = get_last_books_csv_fname(folder)
	print('Loading '+ key)
	df  = s3_read_csv(key)
	return df

def s3_read_csv(key, index_col = 'Book'):
	""" Load csv from amazon """
	csv_name = key
	if not csv_name.endswith('csv'):
		csv_name = csv_name.strip('.') + '.csv'

	print('Loading '+ csv_name)
	resource = boto3.resource('s3') 
	data_bucket = resource.Bucket('datascience.unbound.com')
	obj = data_bucket.Object(csv_name)
	f = obj.get()['Body'].read()
	df = pd.read_csv(io.StringIO(f.decode('utf-8')),index_col=index_col)
	return df

def s3_df_to_csv(key, df):
	csv_name = key
	if not csv_name.endswith('csv'):
		csv_name = csv_name.strip('.') + '.csv'

	resource = boto3.resource('s3') 
	data_bucket = resource.Bucket('datascience.unbound.com')
	f_obj = data_bucket.Object(csv_name)
	f_obj.put(Body = df.to_csv())

def s3_read_json(key):
	if not key.endswith('json'):
		key = key.strip('.') + '.json'
	print('Loading '+ key)
	resource = boto3.resource('s3') 
	data_bucket = resource.Bucket('datascience.unbound.com')
	obj = data_bucket.Object(key)
	f = obj.get()['Body'].read().decode('utf-8')
	return (json.loads(f))

def s3_obj_to_json(key, obj):
	if not key.endswith('json'):
		key = key.strip('.') + '.json'
	resource = boto3.resource('s3') 
	data_bucket = resource.Bucket('datascience.unbound.com')
	new_json_obj = data_bucket.Object(key)
	new_json_obj.put(Body = json.dumps(obj))

def s3_unpickle_object(key):
    if not key.endswith('.p'):
        key = key.strip('.') + '.p'
    resource = boto3.resource('s3') 
    data_bucket = resource.Bucket('datascience.unbound.com')
    read_obj = data_bucket.Object(key)
    f = read_obj.get()['Body'].read()#.decode('utf-8')
    return p.loads(f)

def s3_pickle_object(key, obj):
    if not key.endswith('.p'):
        key = key.strip('.') + '.p'
    resource = boto3.resource('s3') 
    data_bucket = resource.Bucket('datascience.unbound.com')
    write_obj = data_bucket.Object(key)
    write_obj.put(Body = p.dumps(obj))
    print(key)