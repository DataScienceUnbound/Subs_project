import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer, MultiLabelBinarizer
from sklearn.utils.multiclass import type_of_target

import sys

from sklearn.preprocessing import MultiLabelBinarizer

def select_column_subset_for_model(df):
    '''
    Selects model relvant columns from books.csv file
    
    Returns dataframe with only the following columns in it:
    
    'Book', 'Title', 'Author_twitter_handle',
    'Super_genres', 'Tags','Digital_first','Submission_or_commissioned',
    'Author_genders', 'Result', 'days_until_funded',
    'days_until_failed', 'Revenue','Revenue_120_days','Revenue_30_days','Revenue_7_days',
    'Book_live_date', 'Book_failed_date','Target', 
    'Book_funded_date','Authors'
    
    '''
    pd_friendly_columns = [col_string.replace(' ', '_') for col_string in df.columns]
    df.columns = pd_friendly_columns
    
    basic_model_cols = ['Book', 'Title', 'Author_twitter_handle',
                        'Super_genres', 'Tags','Digital_first','Submission_or_commissioned',
                        'Author_genders', 'Result', 'days_until_funded',
                        'days_until_failed', 'Revenue','Revenue_120_days','Revenue_30_days','Revenue_7_days',
                        'Book_live_date', 'Book_failed_date','Target', 
                        'Book_funded_date','Authors']
    
    # below is to handle change to books csv file that was made in the back end
    if 'Book_public_date' in df.columns:
        df = df.rename(columns={'Book_public_date':'Book_live_date'})
        
    drop_cols = [col for col in df.columns if col not in basic_model_cols]
    df = df.drop(drop_cols, axis =1)
    return df
    


def convert_tag_series_to_binarised_df(series):
    """
    This function converts a pandas series of tag lists into a dataframe. Each column
    contains 1's and 0's referring to whether if the column tagname was in
    the series.

    TODO: explain this better

    """
    mlb = MultiLabelBinarizer()
    mlb_frame = pd.DataFrame(data=mlb.fit_transform(series.dropna()),
                             columns=mlb.classes_,
                             index=series.dropna().index)

    null_frame = pd.DataFrame(series[series.isnull()])
    for col in mlb.classes_:
        null_frame[col] = np.nan
    null_frame.drop(series.name, axis=1, inplace=True)

    binarised_df = pd.concat([null_frame, mlb_frame])
    binarised_df = binarised_df.sort_index()

    print('Binarised series into df with ' + str(len(mlb.classes_)) + ' columns:', mlb.classes_)

    if series.isnull().sum():
        print('***** WARNING! *****')
        print('There were ' + str(series.isnull().sum()) + ' rows without data.')

    return binarised_df , mlb 



def prepare_taglike_series(series):
    """
    This function converts a series of tag-strings into a series of lists that
    contain the tags, stripped of whitespace and lower-case.

    Arguments:
    - pandas series whose items contain strings of tags separated by commas.

    Returns:
    - pandas series in which items are lists of tags.
    """
    if type(series.iloc[0]) == list:
        print('Series entries are already lists, assuming already converted - not running')
        return series
    else:
        for i, row_entry in series.iteritems():
            if not pd.isnull(row_entry):
                try:
                    row_entry = row_entry.split(',')
                    series.loc[i] = [tag.strip().lower() for tag in row_entry]
                except Exception as e:
                    print(e)

        return series





def use_tags_to_complete_genres(row, valid_genres):
    """
    This function is takes a row of the books dataframe and produces a
    genre list that contains and valid genres contained in the tags column.

    Arguments:
    - books dataframe row. e.g. i, row = df.iterrows(). Row must contain 'Tags' and
    'Super_genres' columns that both contain lists of tags.
    - set of valid genres

    Returns:
    - genres list
    - tags list

    Example:
    genre_set = set(genres)
    for i, book_row in df.iterrows():
        new_genres, new_tags = use_tags_to_complete_genres(book_row, valid_genres=genre_set)
        #df = df.set_value(i, 'Super_genres', new_genres)
        #df = df.set_value(i, 'Tags', new_tags)
        if new_genres != book_row.Super_genres:
            pass
            #print('hit', i)
            #new_genres,book_row.Super_genres
    """
    # i'm not actually sure if we need to copy
    if row.Tags is np.NaN:
        tags = np.NaN
    else:
        tags = row.Tags.copy()

    if row.Super_genres is np.NaN:
        genres = np.NaN
    else:
        genres = row.Super_genres.copy()

    # nothing we can do if no tags
    if tags is np.NaN:
        return genres, tags

    # search through tags looking if tag is found in genres
    for tag in tags:
        if tag in valid_genres:
            # as we are moving it to genres don't need it in tags
            # tags.remove(tag)

            # add tag to the genres list if not there
            if genres is np.NaN:
                genres = [tag]
            elif tag not in set(genres):
                genres.append(tag)
            else:
                pass

    return genres, tags

def get_taglike_series_info(series, print_info=True, verbose=True):
    """
    This function returns info of taglike series of lists, e.g unique tags and their freq.

    Arguments:
    - series: pandas series object whose items are lists of tags.
    - print_info: flag to print info of tag series
    - verbose: flag to print all of the unique tags or just 200 chars

    Returns:
    - unique tags list
    - tag_freq_dict: tag to tag-frequency dictionary
    - tag_freqs: sorted series containing tag_name as index and corresponding
      frequency of tag occurance as value.
    """

    tag_freq_dict = {}
    for i,row_list in series.iteritems():
        if row_list is not np.NaN:
            for tag_name in row_list:
                tag_freq_dict[tag_name] = tag_freq_dict.get(tag_name, 1) + 1

    unique_tags = list(tag_freq_dict.keys())
    tag_freqs   = sorted(tag_freq_dict.items(), key=lambda x: x[1], reverse=True)
    # tag_freq_dict.items() is a list of tuples, each tuple is a (tag, freq) pair.
    # key=lambda x:x[1] sorts on the second index (frequency) of the tuple.

    tag_freqs = np.array(tag_freqs)
    tag_freqs = pd.Series(index=tag_freqs[:,0], data=tag_freqs[:,1].astype(int))

    if print_info:
        if verbose: i = -1 # print everthing
        else: i = 200
        print(str(len(unique_tags))+ ' unique tags found in taglike series, printing 1st '+str(i)+ ' chars:')
        print(tag_freqs.__repr__()[:i])

    return unique_tags, tag_freq_dict, tag_freqs

def load_dataframe(pathname, azure=False):
    '''
    Returns dataframe given a pathname. Currently just expecting to be pointed
    to an absolute local path, but function will be changed to handle azure in future.

    Arguments:
        - pathname: absolute path if azure is False

    Returns:
        - pandas dataframe
    '''

    df = pd.read_csv(pathname)

    if azure:
        print('Azure functionality not implemented yet')
    return df

def get_nominal_series_dummy_df_old(series):
    """
    Converts a nominal series into a dataframe with dummy variable columns with 'intelligent' handling of colnames.

    Arguments:
    - series of nominal data.

    Returns:
    - dataframe of dummy variables for series.
    """
    if series.isnull().sum():
        raise Exception('Error: series contains NaN')

    lb = LabelBinarizer()
    if type_of_target(series) == 'binary':
        if series.dtype == 'object':
            binarised_series = pd.Series(lb.fit_transform(series).ravel(), index=series.index)
            binarised_series.name = '0:'+lb.classes_[0]+' 1:'+lb.classes_[1]
            dummy_df = pd.DataFrame(binarised_series, index=series.index)
        else:
            print('WARNING: Not changing series:'+series.name+' as non-string binary feature')
            dummy_df = pd.DataFrame(series, index = series.index)

    elif type_of_target(series) == 'multiclass':
        dummy_df = pd.DataFrame(lb.fit_transform(series), index = series.index)
        if series.dtype == 'object':
            dummy_df.columns =  lb.classes_
        else:
            col_names = [series.name+': '+str(level) for level in lb.classes_]
            dummy_df.columns =  col_names

        # check counts are the same
        old_counts = series.value_counts().sort_values(ascending=False)
        new_counts = dummy_df.sum(axis = 0).sort_values(ascending=False)
        assert np.array_equal(new_counts.values, old_counts.values)

    return dummy_df, lb 


def get_nominal_series_dummy_df(series):
    """
    Converts a nominal series into a dataframe of dummies.
    Attemtps to handle column names and strings well.

    Arguments:
    - series of nominal data.

    Returns:
    - dataframe of dummy variables for series.
    """
    if series.isnull().sum():
        raise Exception('Error: series contains NaN')

    lb = LabelBinarizer()
    if type_of_target(series) == 'binary':
        if series.dtype == 'object':
            binarised_series = pd.Series(lb.fit_transform(series).ravel(), index=series.index)
            binarised_series.name = '0:' + lb.classes_[0] + ' 1:' + lb.classes_[1]
            dummy_df = pd.DataFrame(binarised_series, index=series.index)
            lb.unbound_model_df_colnames = [binarised_series.name]
        else:
            print('WARNING: Not changing series:' + series.name + ' as non-string binary feature')
            dummy_df = pd.DataFrame(series, index=series.index)
            lb.unbound_model_df_colnames = [series.name]

    elif type_of_target(series) == 'multiclass':
        dummy_df = pd.DataFrame(lb.fit_transform(series), index=series.index)
        if series.dtype == 'object':
            dummy_df.columns = lb.classes_
        else:
            col_names = [series.name + ': ' + str(level) for level in lb.classes_]
            dummy_df.columns = col_names
        lb.unbound_model_df_colnames = dummy_df.columns

        # check counts are the same
        old_counts = series.value_counts().sort_values(ascending=False)
        new_counts = dummy_df.sum(axis=0).sort_values(ascending=False)
        assert np.array_equal(new_counts.values, old_counts.values)

    # print(dummy_df.isnull().sum())
    return dummy_df, lb

if __name__ == "__main__":
    pass