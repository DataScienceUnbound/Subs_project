import os
import sys
import datetime

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import json 
import io
import tweepy
#import skopt

# cull these imports!
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as sk_metrics
from sklearn.metrics import accuracy_score, f1_score, fowlkes_mallows_score 
import sklearn.model_selection as sk_model_selection 
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ARDRegression, Ridge, ElasticNet,Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor

# added by ML
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
import scipy.stats as stats

import data_preparation as data_utils
import s3_utils

from twitter_data_utils_v2 import make_parsed_dict, TwitterAnalyser

from tweet_parser.tweet import Tweet
from searchtweets import (ResultStream,
                           collect_results,
                           gen_rule_payload,
                           load_credentials)

class RevenueModel():
    """
    Not tested yet on a dataframe, just passing in the series.
    """

    def __init__(self):
        # info fields - not actually using?
        self.books_csv_training_set_name = None
        self.required_fields = None
        self.optional_fields = None

        # label binerisers and pca objects from notebook 1
        self.gender_mlb = None #
        self.genre_mlb = None #
        self.tag_mlb = None #
        self.tag_pca_obj = None #
        self.digital_lb = None #
        self.sub_comissioned_lb = None 
        self.tag_pca_obj = None

        
        # From training prediction model notebook - not sure if using
        self.categorical_pca_cols = None
        self.categorical_pca_object = None

        self.revenue_model_object = None  # this is the object made in the last notebook
        
        # Model preprocessing
        self.std_scaler = None
        self.std_scaler_col_order = None
        
        # The models themselves
        self.revenue_d7_model = None
        self.revenue_d30_model = None
        self.revenue_d120_model = None
        self.mae_7 = None
        self.mae_30 = None
        self.mae_120 = None
        
        # Twitter data specific - split nb1 processing?
        self.api_object = TwitterApiHandler()
        self.parsed_dict = None
        #self.historical_queries = {''} # load/store in api handler?
        self.historical_queries = s3_utils.s3_unpickle_object('twitter_data/api_data_prediction_queries/historical_queryrules.p')
    
    
    def check_fields(self):
        #todo check inputs are in
        print('Gender options:')
        print(self.gender_mlb.classes_)
        print('Super Genre options:')
        print(self.genre_mlb.classes_)
        print('Digital_first options')
        print(self.digital_lb.classes_)
        print('Submission_or_commissioned options')
        print(self.sub_comissioned_lb.classes_)
        print('Tag options:')
        print(self.tag_mlb.classes_)
        
    @staticmethod
    def eval_if_not_list(x):
        '''
        function to be applied to series
        '''
        if type(x) != list:
            try:
                x = eval(x)
            except:
                pass
        return x

    def run_models(self,figure_file_path='', target=10000):
        '''
        
        '''
        model_output = self.revenue_model_object.predict(self.preprocessed_frame,figure_file_path=figure_file_path,target = target)
        return model_output
    
    def run_pca_on_categorical_features(self):
        print('Running PCA on categorical features')
        self.pca_df = self.preprocessed_frame.loc[:, self.categorical_pca_cols].copy()
        pca_components = self.categorical_pca_object.transform(self.pca_df)
        pca_comp_df = pd.DataFrame(pca_components,
                                index = self.pca_df.index,
                                    columns = ['pca_'+str(i) for i in range(self.categorical_pca_object.n_components)])
        self.preprocessed_frame	= self.preprocessed_frame.drop(self.categorical_pca_cols, axis = 1, inplace=False)
        self.preprocessed_frame = pd.concat([self.preprocessed_frame, pca_comp_df], axis = 1)

    def predict_from_series(self, pandas_series,date, days_before_to_scrape,figure_file_path='',target=10000):
        self.df = pd.Series(pandas_series).to_frame().transpose() # make sure pandas_series is a pandas series
        model_output = self.predict_from_df(self.df,date, days_before_to_scrape,figure_file_path,target)
        return model_output

    def predict_from_df(self, df, date, days_before_to_scrape,figure_file_path='',target=10000,log_search = True):
        """
        
        date should be a string eg. '2018-04-29'
        """
        df.loc[:,'Book_live_date'] = date # hacky
        
        # nb1 stiff
        self.process_genre(df)
        self.process_tags(df)
        self.process_gender(df)
        self.process_sub_comissioned(df)
        self.process_digital(df)
        self.process_date(df)
        df_nb1 = self.get_final_nb1_frame(df)
        self.process_twitter_handles(df,date, days_before_to_scrape)
        self.preprocessed_frame = self.scrape_analyser.combine_analysis_with_raw_df(df_nb1)
        self.preprocessed_frame = self.preprocessed_frame.drop('scraped_authors', axis = 1)
        model_output = self.run_models(figure_file_path,target)
        #Log searches
        if log_search:
            search = {'Input': df,'Output':model_output}
            logfname = str(str([df.twitter_author_lists[i]+['_'] for i in range(len(df.twitter_author_lists[:]))]))
            logfname = logfname.replace('[','').replace(']','').replace('\'','').replace(',','').replace(' ','') + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            s3_utils.s3_pickle_object('revenue_models_v2/p logs/'+logfname, search)
        
        return model_output

    def process_genre(self, df):
        self.mlb_genre_frame = pd.DataFrame(data=self.genre_mlb.transform(df.Super_genres),
                                 columns =self.genre_mlb.classes_,
                                 index = df.index)

    def process_tags(self, df):
        mlb_tag_frame = pd.DataFrame(data=self.tag_mlb.transform(df.Tags),
                                          columns =self.tag_mlb.classes_,
                                          index = df.index)
        self.tags_pca_frame = pd.DataFrame(data=self.tag_pca_obj.transform(mlb_tag_frame),
                                           index = df.index, 
                                           columns = ['tag_pca_'+str(i) for i in range(self.tag_pca_obj.n_components)])
    def process_gender(self, df):
        self.mlb_gender_frame = pd.DataFrame(data=self.gender_mlb.transform(df.Author_genders),
                                             columns =self.gender_mlb.classes_,
                                             index = df.index)
    def process_sub_comissioned(self,df):
        self.sub_comissioned_frame = pd.DataFrame(data=self.sub_comissioned_lb.transform(df.Submission_or_commissioned),
                                                  columns =self.sub_comissioned_lb.classes_,
                                                  index = df.index)
    def process_digital(self,df):
        self.digital_first_frame = pd.DataFrame(data=self.digital_lb.transform(df.Digital_first),
                                                  columns =self.digital_lb.unbound_model_df_colnames,
                                                  index = df.index)
    
    def process_date(self,df,harmonics=3):
        dayofyear = pd.to_datetime(df['Book_live_date']).dt.dayofyear
        self.season_frame = pd.concat(
            [pd.DataFrame(data = np.array([np.cos((i+1)*2*np.pi*dayofyear/366), np.sin((i+1)*2*np.pi*dayofyear/366)]).T,
                          index=df.index,columns=['season_cos'+str(i+1),'season_sin'+str(i+1)]) 
             for i in range(harmonics) ],
            axis=1)


        
    def get_final_nb1_frame(self,df):
        #self.twitter_frame.index = self.twitter_frame.index.astype(int) # stupid, correct this

        self.preprocessed_frame = pd.concat([self.mlb_genre_frame,
                                             self.tags_pca_frame,
                                             self.mlb_gender_frame,
                                             self.sub_comissioned_frame,
                                             self.digital_first_frame,
                                             self.df.loc[:, ['n_authors', 'n_twitter_authors']],#
                                             self.season_frame,
                                            ],axis = 1)
        return self.preprocessed_frame # change this name
        #here the drops should be saved to model from notebook 4


    def process_twitter_handles(self, df, date, days_before_to_scrape):
        """
        First checks if we have already queried the api for this author
        """
        for book_id, a_list in df.twitter_author_lists.iteritems():
            author_list = self.eval_if_not_list(a_list)
            
            if type(author_list) == list: 
                author_list = [a.lower().strip().strip('@') for a in author_list]
                subtracted_date = self.api_object.subtract_from_datestring(date,
                                                                           days_before_to_scrape)
                for handle in author_list:
                    rule = self.api_object.make_rule(handle,date,subtracted_date,
                                                     results_per_call=500)
                    rule_to_check = self.api_object.strip_maxresults_from_query(rule)
                    if rule_to_check in self.historical_queries:
                        print('move to parsing function/skip')
                        print('rule found')
                    else:
                        # print loading raw scrapes - may take some time
                        self.call_api(handle,date,days_before_to_scrape)
                        #historical_queries.add(rule_to_check) - actuall add only when successfuk
        
        # should be good to move to the fuction that loads parsed dict only
        self.run_scraper_analyser(df,days_before_to_scrape)
    
    def run_scraper_analyser(self,df,days_before_to_scrape):
        if self.parsed_dict is None:
            self.parsed_dict = s3_utils.s3_unpickle_object('twitter_data/api_data_prediction_queries/historical_parsed_dict.p')
        
        self.scrape_analyser = TwitterAnalyser(self.parsed_dict,
                                  days_before = days_before_to_scrape)
        
        self.scrape_analyser.analyse_scrapes(df)
        self.scrape_analyser.combine_authors()
            
    def call_api(self, handle,date,days_before_to_scrape):
        query, results_list = self.api_object.pull_data_for_handle(handle,date, days_before_to_scrape)
        self.historical_queries.add(query)
        s3_utils.s3_pickle_object('twitter_data/api_data_prediction_queries/historical_queryrules.p',
                                  self.historical_queries)
        
        # here save the new raw search
        s3_location_string = 'twitter_data/api_data/raw_data_searches/predicton_queries/'
        query_string = query[1:-1].replace("'", '').replace(',','_').replace(':','_').replace(' ','')
        s3_query_key = s3_location_string+query_string+'.p'
        s3_utils.s3_pickle_object(s3_query_key, results_list)
        
        new_parsed_dict = make_parsed_dict({query:results_list})
        self.update_historical_parsed_dict(new_parsed_dict)
        # here load and save to the master dict
        # if not then call the function to remake the parsed dict
    
    def update_historical_parsed_dict(self, update_dict):
        if self.parsed_dict is None:
            self.parsed_dict = s3_utils.s3_unpickle_object('twitter_data/api_data_prediction_queries/historical_parsed_dict.p')
        self.parsed_dict = {**self.parsed_dict,**update_dict}
        s3_utils.s3_pickle_object('twitter_data/api_data_prediction_queries/historical_parsed_dict.p', self.parsed_dict)

    def get_historical_queries(self, s3_location_string = 'twitter_data/api_data/raw_data_searches/predicton_queries/'):
        '''note we do not use this atm as saving queries into a list historical_queryrules'''
        historical_searches = [os.path.split(k)[1] for k in s3_utils.s3_list(s3_location_string) if k.endswith('.p')]
        return historical_searches
    
    def check_if_query_previously_pulled(self, query, historical_searches):
        '''note we do not use this atm as saving queries into a list historical_queryrules'''
        query_string = query[1:-1].replace("'", '').replace(',','_').replace(':','_').replace(' ','')
        if query_string+'.p' in historical_searches:
            return True
        else:
            return False


class TwitterApiHandler():
    '''
    Class to handle interaction with tweepy and twitter api.
    
    Should probably be used in notebook 2 instead of cells
    '''
    
    def __init__(self):
        self.tweepy_api_credentials_path = 'credentials/api_credentials.json' # add _full.json for the historical
        self.get_tweepy_api_object()
        
        # this should be the 30 day sandbox for the prediction model
        self.enpoint_credientials_path = 'credentials/twitter_keys_full.yaml'# add _full.yaml for the historical
        self.get_endpoint()
    
    def get_endpoint(self):
        '''to be passed in as search args'''
        self.endpoint_args = load_credentials(self.enpoint_credientials_path)
    
    def get_tweepy_api_object(self):
        with open(self.tweepy_api_credentials_path) as f: # this is the full
            tweepy_keys = json.load(f)
            auth = tweepy.OAuthHandler(tweepy_keys['consumer_key'], tweepy_keys['consumer_secret'])
            auth.set_access_token(tweepy_keys['access_token'], tweepy_keys['access_token_secret'])
            self.tweepy_api = tweepy.API(auth)
            
    def get_handle_id(self, handle):
        try:
            user = self.tweepy_api.get_user(handle)
            return user.id_str
        except tweepy.TweepError as e:
            if eval(e.__dict__['reason'])[0]['message'] =='User not found.':
                print('User not found: ', handle)
                return 0
            else:
                print(e)
                return 0
            
    def pull_data_for_handle(self,handle,date,days_before, 
                             results_per_call=100, max_results = 2500):
        # check handle can be found!
        user_id = self.get_handle_id(handle)
        if user_id is 0:
            return 0
        from_date = self.subtract_from_datestring(date, days_before)
        rule = self.make_rule(handle, date, from_date,results_per_call)
        
        rs = ResultStream(rule_payload=rule,
                          max_results=max_results,
                          **self.endpoint_args)
        results_list = list(rs.stream())
#         results_list=temp_dict[list(temp_dict.keys())[0]]
        print('Found', len(results_list),'tweets for',handle)
        if len(results_list) == max_results:
            print('Max results limit hit ('+str(2500)+'). Consider changing the parameter')
                  
        return self.strip_maxresults_from_query(rule), results_list
            
    @staticmethod        
    def make_rule(handle, 
                  to_date, 
                  from_date, 
                  results_per_call):
        """
        Inputs:
            - handle (should be changed to id)
            - to_date
        """
        #print('Using',results_per_call,' results per call. Should be 100 for sandbox, 500 for premium')
        _rule_a = "from:"+handle
        rule = gen_rule_payload(_rule_a,
                            from_date=from_date,
                            to_date=to_date,
                            results_per_call=results_per_call)
        return rule
    
    @staticmethod
    def strip_maxresults_from_query(query):
        query_dict = eval(query)
        del(query_dict['maxResults'])
        return(str(query_dict))
    
    @staticmethod
    def subtract_from_datestring(date, days_to_subtract):
        '''
            - date: string, e.g. '2018-04-17'
            - days_to_subtract: integer
            
        returns:
            - string of date minus days
            e.g. 2018-04-12
        '''
        to_datetime = pd.to_datetime(date)
        from_dateime = to_datetime - pd.Timedelta(days_to_subtract, unit='D')
        from_datestring = str(from_dateime.date())
        return from_datestring

    
def log_convolve_pdes(a,b,drawplots = False): # Need to describe this integral in markdown on cell above
    # get new x axis and transform a and b to that axis
    a[1][np.isnan(a[1])]=0
    b[1][np.isnan(b[1])]=0
    arg_aM = np.log(np.maximum(np.exp(a[0][np.newaxis,:])-np.exp(b[0][:,np.newaxis]),np.exp(a[0][0]))+1)
    M0 = np.interp(arg_aM,a[0],a[1])
    M0 = M0*((np.exp(a[0][np.newaxis,:])/(1+np.maximum(np.exp(a[0][np.newaxis,:])-np.exp(b[0][:,np.newaxis]),.1e-12))))
    M0 = M0*np.array([[i>j for i in range(len(a[0]))] for j in range(len(b[0]))])
    M0 = M0 + np.eye(M0.shape[0],M0.shape[1],1)*1e-32
    M0[-1,-1] =1
    M0 = M0/(np.sum(M0,axis=1)[:,np.newaxis])
    
    fz = b[1].T@M0
    if drawplots:
        plt.figure(figsize=(10,10))
        plt.imshow(M0)
    return (a[0],fz)

def log_sum_pdes(a,b, drawplots = False,n=1000):
    x_icdf   = np.linspace(0,1,n*len(a[0])+1)[1:]
    dp       = x_icdf[1] - x_icdf[0]
    da       = a[0][1]-a[0][0]
    db       = b[0][1]-b[0][0]
    a_icdf   = np.interp(x_icdf,np.cumsum(a[1])*da,a[0]) # inverse cumulative distribution of a
    b_icdf   = np.interp(x_icdf,np.cumsum(b[1])*db,b[0]) # inverse cumulative distribution of a
    cx_icdf  = np.log(np.minimum(np.maximum(np.exp(a_icdf) + np.exp(b_icdf) - 1,1e-4),1e16)) # assume worst cas scenario of perfect rank correlation between pdfs
    dcx_icdf = .5*(np.hstack((0,np.diff(cx_icdf)))+ np.hstack((np.diff(cx_icdf),0)))+1e-12
    c_pdf    = np.interp(a[0],cx_icdf,dp/dcx_icdf)
    c_pdf    = c_pdf - c_pdf[-1] # regularize bins under the percision limit of difference operation
    c_pdf[c_pdf<=0] = 0
    c_pdf    = c_pdf/(np.sum(c_pdf)*da) # regularize integral
    return (a[0],c_pdf)
    
def get_CI_from_posterior(a,CI):
    a_cdf = np.cumsum(a[1])*(a[0][1]-a[0][0])
    ci_list = []
    for ci in CI:
        ci_list.append((max(a[0][a_cdf<.5-ci/2]),min(a[0][a_cdf>(.5+ci/2)])))
    return np.array(ci_list)



class RankLogisticDoubleRegression:
    def __init__(self,skmodel0 = LinearRegression(),skmodel1 = LinearRegression(),poly_degree=1,class_threshold = .85,overweight = 2,preselected_features = []):
        self.model0 = skmodel0
        self.model1 = skmodel1
        self.poly = PolynomialFeatures(degree=poly_degree) # polynomial expansion of the features
        self.class_threshold = class_threshold
        self.overweight = overweight
        self.preselected_features=preselected_features
        self.z50  = norm.ppf(.25)
        self.z95  = norm.ppf(.05) # for 90% Confidence interval and not 95!!! (for 95 use .025 instead of .05!)
        
    def maxrank1norm(self,x):
        return x/(np.max(x)+1)
    def diff2(self,x):
        return .5*(np.hstack((0,np.diff(x)))+ np.hstack((np.diff(x),0)))

    def fit(self,x_train,y_train):
        
        if not(len(self.preselected_features)==0):
            x_train = x_train[self.preselected_features] 
        
        self.x_train = pd.DataFrame(self.poly.fit_transform(x_train))._get_numeric_data() #polynomial expansion of features
        x_train_r = self.x_train.rank()
        x_train_z = x_train_r.apply(self.maxrank1norm).applymap(norm.ppf) # compute equivalent Z value
        self.x_train_z = x_train_z
        
        self.y_train = pd.DataFrame(y_train)._get_numeric_data()
        y_train_r = self.y_train.rank()
        self.y_train_z = y_train_r.apply(self.maxrank1norm).applymap(norm.ppf) # compute equivalent Z value and save for making predictions
      
        self.logreg = LogisticRegression()
        
        class0_ind = self.maxrank1norm(y_train_r.get_values().ravel()) <= self.class_threshold
        class1_ind = self.maxrank1norm(y_train_r.get_values().ravel()) > self.class_threshold
        weights = (class1_ind*self.overweight+1)
        weights = weights.ravel()
        self.logreg.fit(x_train_z,class1_ind,sample_weight=weights)# Use weights are to give 3x more importance in finding the good books
        
        #Consider using the following labels for eps estimation...
        class0_ind_eps = self.logreg.predict_proba(x_train_z)[:,0]>=.5
        class1_ind_eps = self.logreg.predict_proba(x_train_z)[:,0]<.5
    
        self.model0.fit(x_train_z[class0_ind],self.y_train_z[class0_ind]) 
        self.model1.fit(x_train_z[class1_ind],self.y_train_z[class1_ind])
        
        eps0 = self.model0.predict(x_train_z[class0_ind_eps])-self.y_train_z[class0_ind_eps].T
        eps1 = self.model1.predict(x_train_z[class1_ind_eps])-self.y_train_z[class1_ind_eps].T
        self.eps_std0 = np.sqrt(np.mean(eps0.get_values()**2))
        self.eps_std1 = np.sqrt(np.mean(eps1.get_values()**2))
        return self
        
    def predict(self, xvalv, predict_CI=[],save_posteriors = False, drawplots = False, output_scores=False):
        if not(len(self.preselected_features)==0):
            xvalv = xvalv[self.preselected_features] 
        xvalv = pd.DataFrame(self.poly.fit_transform(xvalv))
        xvalv = xvalv.get_values()
        yval = []
        yval_CI = []
        yval_posteriors = []
        for xvali in range(xvalv.shape[0]):
            xval = xvalv[xvali,:]
            xval_r = (np.nanmean(self.x_train.get_values()<xval,axis=0) 
                     + 0.5*np.nanmean(self.x_train.get_values()==xval,axis=0)) # rank feature vector relative to training data
            #print('clipping from below:',self.x_train.columns[xval_r==0])
            xval_r[xval_r==0] = 1/(self.x_train.shape[0]+1) # clip rankings to avoid out of bound infinities
            #print('clipping from above:',self.x_train.columns[xval_r==1])
            xval_r[xval_r==1] = 1-1/(self.x_train.shape[0]+1)
            xval_z = norm.ppf(xval_r) # compute equivalent Z value
            xval_z = xval_z.reshape(1, -1)
            
            label = self.logreg.predict_proba(xval_z)
            predict0 = self.model0.predict(xval_z)
            predict1 = self.model1.predict(xval_z)
            yval_z = predict0*label[:,0] + predict1*label[:,1]
            eps = np.sqrt(label[:,0]*self.eps_std0**2 + label[:,1]*self.eps_std1**2) # pretend both predictions are independent and just average variances
            
            ypercentile = norm.cdf(yval_z)
            yval_arg = np.argmin(np.abs(self.y_train_z.get_values() - yval_z))
            yval.append(self.y_train.get_values()[yval_arg])
            
            if not(predict_CI==[]):
                y_axis_values = np.linspace(-2,16,500) # reasonable limmits for book log revenues
                dy = y_axis_values[1]-y_axis_values[0]
                KDE    = stats.gaussian_kde(self.y_train.get_values().ravel(), bw_method= 2/np.sqrt(len(self.y_train)))
                y_pdf  = KDE.pdf(y_axis_values)
                y_cdf  = np.cumsum(y_pdf)*dy
                yz     = norm.ppf(y_cdf) # function that goes from Y to Z space
                posterior_yz = np.exp(-.5*((yz-yval_z)/eps)**2)/np.sqrt(2*np.pi*eps**2)
                posterior_y = posterior_yz*self.diff2(yz)/dy # Change of variables between Z space and Y space trhough the function yz (d(yz)/dy is always positive)
                posterior_y[np.logical_not(np.isfinite(posterior_y))] = 0
                posterior_y = posterior_y/(np.sum(posterior_y)*dy) # correct numerical errors and make sure pdf sums to one 
                posterior_y_cdf = np.cumsum(posterior_y)*dy
                expected_revenue = posterior_y.dot(np.exp(y_axis_values)-1)
                
                ci_list = []
                for ci in predict_CI:
                    ci_list.append((max(y_axis_values[posterior_y_cdf<.5-ci/2]),min(y_axis_values[posterior_y_cdf>(.5+ci/2)])))
                yval_CI.append(ci_list)
                yval_posteriors.append((y_axis_values,posterior_y))
                
                if drawplots:
                    #Drawing plots also implies computing full posterior distributions
                    #compute change of variables from posterior Zs to posterior Log Revenues

                    plt.hist(self.y_train.get_values(),bins=int(4*np.ceil(np.sqrt(self.y_train.shape[0]))),normed=True,alpha=.2)
                    plt.plot(y_axis_values,y_pdf,color=[.2,.2,.7],linewidth=3)
                    plt.hlines([-.05],ci_list[1][0],ci_list[1][1],'r',linewidth=3,alpha=.5) # plot CIs of prediction
                    plt.hlines([-.05],ci_list[0][0],ci_list[0][1],'r',linewidth=5,alpha=.8)  
                    plt.vlines(yval[-1],-.05,-.025,'r',linewidth=3)
                    plt.plot(y_axis_values,posterior_y,color=[1,0,0])
                    plt.hlines([0],y_axis_values[1],y_axis_values[-1],'k',linewidth=1,alpha=1)
                    plt.xticks([0,np.log(10),np.log(100),np.log(1000),np.log(10000),np.log(100000)],['1','10','100','1k','10k','100k'])

        if not(predict_CI==[]):
            if save_posteriors:
                if output_scores:
                    scores = label[:,0]*self.model0.coef_*xval_z + label[:,1]*self.model0.coef_*xval_z
                    scores=pd.DataFrame(scores.ravel()[1:],index=self.preselected_features)
                    return np.array(yval).ravel(), np.array(yval_CI), yval_posteriors, scores
                else:
                    return np.array(yval).ravel(), np.array(yval_CI), yval_posteriors
            else:
                return np.array(yval).ravel(), np.array(yval_CI)
        else:
            return np.array(yval).ravel()
        
    
class CombinedModel():
    '''
    Class to output model predictios
    '''
    def __init__(self,std_scaler,std_scaler_col_order,rev7,rev30,rev120,y7,y30,y120,version):
        self.std_scaler = std_scaler
        self.std_scaler_order = std_scaler_col_order
        self.rev7 = rev7
        self.rev30 = rev30
        self.rev120 = rev120
        self.y7 = y7
        self.y30 = y30
        self.y120 = y120
        self.version = version 
    
    def log_convolve_pdes(self,a,b,drawplots = False): # Need to describe this integral in markdown on cell above
        # get new x axis and transform a and b to that axis
        a[1][np.isnan(a[1])]=0
        b[1][np.isnan(b[1])]=0
        arg_aM = np.log(np.maximum(np.exp(a[0][np.newaxis,:])-np.exp(b[0][:,np.newaxis]),np.exp(a[0][0]))+1)
        M0 = np.interp(arg_aM,a[0],a[1])
        M0 = M0*((np.exp(a[0][np.newaxis,:])/(1+np.maximum(np.exp(a[0][np.newaxis,:])-np.exp(b[0][:,np.newaxis]),.1e-12))))
        M0 = M0*np.array([[i>j for i in range(len(a[0]))] for j in range(len(b[0]))])
        M0 = M0 + np.eye(M0.shape[0],M0.shape[1],1)*1e-32
        M0[-1,-1] =1
        M0 = M0/(np.sum(M0,axis=1)[:,np.newaxis])

        fz = b[1].T@M0
        if drawplots:
            plt.figure(figsize=(10,10))
            plt.imshow(M0)
        return (a[0],fz)
    
    
    def log_sum_pdes(self,a,b, drawplots = False,n=100):
        x_icdf   = np.linspace(0,1,n*len(a[0])+1)[1:]
        dp       = x_icdf[1] - x_icdf[0]
        da       = a[0][1]-a[0][0]
        db       = b[0][1]-b[0][0]
        a_icdf   = np.interp(x_icdf,np.cumsum(a[1])*da,a[0]) # inverse cumulative distribution of a
        b_icdf   = np.interp(x_icdf,np.cumsum(b[1])*db,b[0]) # inverse cumulative distribution of a
        cx_icdf  = np.log(np.minimum(np.maximum(np.exp(a_icdf) + np.exp(b_icdf) - 1,1e-4),1e16)) # assume worst cas scenario of perfect rank correlation between pdfs
        dcx_icdf = .5*(np.hstack((0,np.diff(cx_icdf)))+ np.hstack((np.diff(cx_icdf),0)))+1e-12
        c_pdf    = np.interp(a[0],cx_icdf,dp/dcx_icdf)
        c_pdf    = c_pdf - c_pdf[-1] # regularize bins under the percision limit of difference operation
        c_pdf[c_pdf<=0] = 0
        c_pdf    = c_pdf/(np.sum(c_pdf)*da) # regularize integral
        return (a[0],c_pdf)
    
    def get_CI_from_posterior(self,a,CI):
        a_cdf = np.cumsum(a[1])*(a[0][1]-a[0][0])
        ci_list = []
        for ci in CI:
            ci_list.append((max(a[0][a_cdf<.5-ci/2]),min(a[0][a_cdf>(.5+ci/2)])))
        return np.array(ci_list)
    
    def predict(self, x, predict_CI=[.5,.9], save_posteriors=True, drawplots=True, target = 10000,figure_file_path = 'plots.svg'):
       
        x_std_scale = pd.DataFrame(self.std_scaler.transform(x.loc[:,self.std_scaler_order]),columns = x.columns,index = x.index)
        #print(x_std_scale)
        str7 = ''
        str30 = ''
        str120 = ''
        
        if not(predict_CI==[]):
            predict_CI = [0] # override empty CIs
            
        if not(predict_CI==[]):
             
            y7_preds, y7_preds_CI, y7_posteriors, socres7         = self.rev7.predict(x_std_scale,predict_CI=[.5,.9],save_posteriors=True,output_scores=True) 
            y30_preds, y30_preds_CI, y30_posteriors, socres30     = self.rev30.predict(x_std_scale,predict_CI=[.5,.9],save_posteriors=True,output_scores=True) 
            y120_preds, y120_preds_CI, y120_posteriors, scores120 = self.rev120.predict(x_std_scale,predict_CI=[.5,.9],save_posteriors=True,output_scores=True)
            
            # average scores
            scores_total = ((np.exp(y7_preds)-1)*socres7 + (np.exp(y30_preds)-1)*socres7 + (np.exp(y120_preds)-1)*socres7)/(np.exp(y7_preds)+np.exp(y30_preds)+np.exp(y120_preds)-3)
            scores_total = scores_total[scores_total.index.str.contains('n_')]
            
            # Just a filter to make the plots prettier 
            N = len(y7_posteriors[0][0])
            kernel = np.array([1,.5,.25])
            kernel = kernel/sum(kernel)
            print(kernel)
            Mfilter_pdf = kernel[0]*np.eye(N,k=0) +.5*kernel[1]*(np.eye(N,k=1) + np.eye(N,k=-1)) +.5*kernel[2]*(np.eye(N,k=2) + np.eye(N,k=-2)) 
            
            y30_posteriors = self.log_sum_pdes(y7_posteriors[0],y30_posteriors[0])
            y120_posteriors = self.log_sum_pdes(y30_posteriors,y120_posteriors[0])
            
            #compute probabilities of reaching target
            logtarget = np.log(target+1)
            dx7= y7_posteriors[0][0][1]-y7_posteriors[0][0][0]
            ptarget7 = int(np.sum(y7_posteriors[0][1][y7_posteriors[0][0]>logtarget])*dx7*100)
            dx30= y30_posteriors[0][1]-y30_posteriors[0][0]
            ptarget30 = int(np.sum(y30_posteriors[1][y30_posteriors[0]>logtarget])*dx30*100)
            dx120= y120_posteriors[0][1]-y120_posteriors[0][0]
            ptarget120 = int(np.sum(y120_posteriors[1][y120_posteriors[0]>logtarget])*dx120*100)
            
            # compute confidence intervals
            y30_preds_CI = self.get_CI_from_posterior(y30_posteriors,[0,.5,.9])
            y120_preds_CI = self.get_CI_from_posterior(y120_posteriors,[0,.5,.9])
            
            
            pred7   = y7_preds
            pred7_h95 = [y7_preds_CI[0,1,1] ]
            pred7_l95 = [y7_preds_CI[0,1,0] ]
            pred7_h50 = [y7_preds_CI[0,0,1] ]
            pred7_l50 = [y7_preds_CI[0,0,0] ]

            pred30   = np.mean(y30_preds_CI[0,:])
            pred30_h95 = [y30_preds_CI[2,1] ]
            pred30_l95 = [y30_preds_CI[2,0] ]
            pred30_h50 = [y30_preds_CI[1,1] ]
            pred30_l50 = [y30_preds_CI[1,0]]
            
            pred120   = np.mean(y120_preds_CI[0,:])
            pred120_h95 = [y120_preds_CI[2,1] ]
            pred120_l95 = [y120_preds_CI[2,0] ]
            pred120_h50 = [y120_preds_CI[1,1] ]
            pred120_l50 = [y120_preds_CI[1,0] ]
            
            str7 = (' (' + str(int(np.exp(pred7_l95[0])-1)) + ' ; ' + str(int( np.exp(pred7_l50[0])-1)) + 
                 ' -  ' + str(int( np.exp(pred7_h50[0])-1)) + ' ; ' + str(int( np.exp(pred7_h95[0])-1)) + ')')
            str30 = ('(' + str(int( np.exp(pred30_l95[0])-1)) + ' ; ' + str(int( np.exp(pred30_l50[0])-1)) + 
                 '  -  ' + str(int( np.exp(pred30_h50[0])-1)) + ' ; ' + str(int( np.exp(pred30_h95[0])-1)) + ')')
            str120 = (' (' + str(int( np.exp(pred120_l95[0])-1)) + ' ; ' + str(int( np.exp(pred120_l50[0])-1)) + 
                 '  -  ' + str(int( np.exp(pred120_h50[0])-1)) + ' ; ' + str(int( np.exp(pred120_h95[0])-1)) + ')')
            
            model_output = {'Predictions':[int(np.exp(pred7[0])-1),int(np.exp(pred30)-1),int(np.exp(pred120)-1)],
                            'Confidence_Intervals':[np.exp(y7_preds_CI[0])-1,np.exp(y30_preds_CI[1:])-1,np.exp(y120_preds_CI[1:])-1],
                            'Posterior_Distributions':[y7_posteriors,y30_posteriors,y120_posteriors],
                            'Probability_of_Reaching_Target':[ptarget7,ptarget30,ptarget120],
                            'Version': self.version }

            if drawplots:
                
                KDE7    = stats.gaussian_kde(np.log(self.y7.get_values().ravel()+1), bw_method= 2/np.sqrt(len(self.y7)))
                y7_pdf  = KDE7.pdf(y7_posteriors[0][0])
                
                KDE30    = stats.gaussian_kde(np.log(self.y30.get_values().ravel()+1), bw_method= 2/np.sqrt(len(self.y30)))
                y30_pdf  = KDE30.pdf(y30_posteriors[0])
                
                KDE120    = stats.gaussian_kde(np.log(self.y120.get_values().ravel()+1), bw_method= 2/np.sqrt(len(self.y120)))
                y120_pdf  = KDE120.pdf(y120_posteriors[0])
                
                model_output['Prior_Distributions'] = [y7_pdf,y30_pdf,y120_pdf]
                
                plt.figure(figsize = (20,13))
                
                plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
                ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2)
                plt.sca(ax1)
                ci_location = 0
                plt.title('Prediction at 7 days\n')
                plt.hist(np.log(self.y7+1),bins=int(2*np.ceil(np.sqrt(self.y7.shape[0]))),normed=True,alpha=.2)
                plt.plot(y7_posteriors[0][0],y7_pdf,color=[.2,.2,.7],linewidth=3,label = 'Books on platform')
                plt.hlines([ci_location-.05],pred7_l95,pred7_h95,'r',linewidth=3,alpha=.5) # plot CIs of prediction
                plt.hlines([ci_location-.05],pred7_l50,pred7_h50,'r',linewidth=5,alpha=.8) 
                plt.vlines(pred7,ci_location-.05,ci_location-.025,'r',linewidth=3) 
                plt.text((pred7_l95[0] + pred7_h95[0])/2,ci_location-.1,'90% confidence interval',alpha=.5,color='r',horizontalalignment='center')
                plt.text((pred7_l50[0] + pred7_h50[0])/2,ci_location-.075,'50% confidence interval',alpha=.8,color='r',horizontalalignment='center')
                plt.text(pred7,ci_location-.02,'prediction: ' + str(int(np.exp(pred7[0])-1))+'£',color='r',horizontalalignment='center')
                plt.plot(y7_posteriors[0][0],y7_posteriors[0][1],color=[1,0,0],label = 'Predicted revenue')
                plt.hlines([0],y7_posteriors[0][0][1],y7_posteriors[0][0][-1],'k',linewidth=1,alpha=1)
                plt.xticks([0,np.log(10),np.log(10**2),np.log(10**3),np.log(10**4),np.log(10**5),np.log(10**6)],['£0','£10','£100','£1k','£10k','£100k','£1m'])
                plt.xlabel('Revenue \n\n Book expected to outperform '+str(int(np.mean(np.less(np.log(self.y7+1),pred7)*100))) +
                           '% of the boooks on the platform\n With a probability of reaching target(' + str(target)+') of ' + str(ptarget7)+'% in 7 days')
                plt.ylabel('Probability density')
                plt.xlim(-2,16)
                plt.ylim(-.125,.5)
                plt.legend(loc=1)


                
                ax2 = plt.subplot2grid((3, 3), (0, 1), rowspan=2)
                plt.sca(ax2)
                ci_location = 0
                plt.title('Prediction at 30 days\n')
                plt.hist(np.log(self.y30+1),bins=int(2*np.ceil(np.sqrt(self.y30.shape[0]))),normed=True,alpha=.2)
                plt.plot(y30_posteriors[0],y30_pdf,color=[.2,.2,.7],linewidth=3,label = 'Books on platform')
                plt.hlines([ci_location-.05],pred30_l95,pred30_h95,'r',linewidth=3,alpha=.5) # plot CIs of prediction
                plt.hlines([ci_location-.05],pred30_l50,pred30_h50,'r',linewidth=5,alpha=.8)  
                plt.vlines(pred30,ci_location-.05,ci_location-.025,'r',linewidth=3)
                plt.text((pred30_l95[0] + pred30_h95[0])/2,ci_location-.1,'90% confidence interval',alpha=.5,color='r',horizontalalignment='center')
                plt.text((pred30_l50[0] + pred30_h50[0])/2,ci_location-.075,'50% confidence interval',alpha=.8,color='r',horizontalalignment='center')
                plt.text(pred30,ci_location-.02,'prediction: ' + str(int(np.exp(pred30)-1))+'£',color='r',horizontalalignment='center')
                plt.plot(y30_posteriors[0],Mfilter_pdf@y30_posteriors[1],color=[1,0,0],label = 'Predicted revenue')
                plt.hlines([0],y30_posteriors[0][1],y30_posteriors[0][-1],'k',linewidth=1,alpha=1)
                plt.xticks([0,np.log(10),np.log(10**2),np.log(10**3),np.log(10**4),np.log(10**5),np.log(10**6)],['£0','£10','£100','£1k','£10k','£100k','£1m'])  
                plt.xlabel('Revenue \n\n Book expected to outperform '+str(int(np.mean(np.less(np.log(self.y30+1),pred30))*100)) +
                           '% of the boooks on the platform\n With a probability of reaching target(' + str(target)+') of ' + str(ptarget30)+'% in 30 days')
                plt.ylabel('Probability density')
                plt.xlim(-2,16)
                plt.ylim(-.125,.5)
                plt.legend(loc=1)
                
                ax3 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
                plt.sca(ax3)
                plt.title('Prediction at 120 days\n')
                plt.hist(np.log(self.y120+1),bins=int(2*np.ceil(np.sqrt(self.y120.shape[0]))),normed=True,alpha=.2)
                plt.plot(y120_posteriors[0],y120_pdf,color=[.2,.2,.7],linewidth=3,label = 'Books on platform')
                plt.hlines([-.05],pred120_l95,pred120_h95,'r',linewidth=3,alpha=.5) # plot CIs of prediction
                plt.hlines([-.05],pred120_l50,pred120_h50,'r',linewidth=5,alpha=.8)  
                plt.vlines(pred120,-.05,-.025,'r',linewidth=3)
                plt.text((pred120_l95[0] + pred120_h95[0])/2,-.1,'90% confidence interval',alpha=.5,color='r',horizontalalignment='center')
                plt.text((pred120_l50[0] + pred120_h50[0])/2,-.075,'50% confidence interval',alpha=.8,color='r',horizontalalignment='center')
                plt.text(pred120,-.02,'prediction: ' + str(int(np.exp(pred120)-1))+'£',color='r',horizontalalignment='center')
                plt.plot(y120_posteriors[0],Mfilter_pdf@y120_posteriors[1],color=[1,0,0],label = 'Predicted revenue')
                plt.hlines([0],y120_posteriors[0][1],y120_posteriors[0][-1],'k',linewidth=1,alpha=1)
                plt.xticks([0,np.log(10),np.log(10**2),np.log(10**3),np.log(10**4),np.log(10**5),np.log(10**6)],['£0','£10','£100','£1k','£10k','£100k','£1m'])
                
                plt.xlabel('Revenue \n\n Book expected to outperform '+str(int(np.mean(np.less(np.log(self.y120+1),pred120))*100)) +
                           '% of the boooks on the platform\n With a probability of reaching target(' + str(target)+') of ' + str(ptarget120)+'% in 120 days')
                plt.ylabel('Probability density')
                plt.xlim(-2,16)
                plt.ylim(-.125,.5)
                plt.legend(loc=1)
                
                
                plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
                plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
                ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
                scores_total.sort_values(by=0).iloc[:-6:-1].append(scores_total.sort_values(by=0).iloc[4::-1]).plot(kind='bar',ax=ax4)
                plt.hlines([0],-1,11,'k',linewidth=1,alpha=1)
                plt.ylabel('Score')
                plt.title('Book most relevant features')
                plt.tight_layout(h_pad=2.0)
                plt.savefig(figure_file_path)
                plt.legend().set_visible(False)
                plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = False
        
                
        else:
            pred7       = self.rev7.predict(x_std_scale, drawplots = drawplots, predict_CI=False) 
            pred30      = self.rev30.predict(x_std_scale, drawplots = drawplots, predict_CI=False) 
            pred120     = self.rev120.predict(x_std_scale, drawplots = drawplots, predict_CI=False) 

#         print('------ prediction ( 5% ; 25%  -  75% ; 95% )')
#         print('Day 7:', int(np.exp(pred7[0])-1), str7)
#         print('Day 30:', int(np.exp(pred30)-1), str30)
#         print('Day 120:', int(np.exp(pred120)-1), str120)
        
        return model_output
    
    
