#!/usr/bin/env python
# coding: utf-8

# A Supreme Court Time Machine

# In[1]:


# Read SCOTUS database

import pandas as pd

cases_df = pd.read_csv('./data/SCDB_2023_01_caseCentered_Docket.csv', encoding='ISO-8859-1')
cases_df


# In[2]:


# Filter SCOTUS Database for relevant/useful columns

cases_selected_columns_df = cases_df[['caseId', 'docketId', 'docket', 'term', 'caseName', 'issueArea', 'partyWinning']]
cases_selected_columns_df.head()


# In[3]:


# Fetching the facts for the SCOTUS data was done outside of this notebook in 
# the "oyez_script/py" file.

SCOTUS_df_unfiltered = pd.read_csv('./data/SCOTUS_data_with_facts')

SCOTUS_df = SCOTUS_df_unfiltered[SCOTUS_df_unfiltered['facts'] != 'Facts of the case not present!']

SCOTUS_df.loc[:, 'partyWinning'] = SCOTUS_df['partyWinning'].astype(bool)

SCOTUS_df


# In[4]:


# Read Kaggle dataset

kaggle_df = pd.read_csv('./data/justice.csv', delimiter=',', encoding = "utf8")
kaggle_df.dropna(inplace=True)
kaggle_df


# In[5]:


# Filter Kaggle dataset for relevant/useful columns

kaggle_selected_columns_df = kaggle_df[['docket', 'term', 'name', 'facts', 'facts_len', 'first_party_winner', 'issue_area']]

kaggle_selected_columns_df.head()
print(kaggle_selected_columns_df.sort_values(by=['term']))


# In[6]:


# Merge SCOTUS and Kaggle database

import numpy as np

# Standardize the docket values to all exist in the format yy-d 
# where (y) is a year digit and (d) is a docket number between 1 and 9999
SCOTUS_df.loc[:, 'docket_standardized_SCOTUS'] = np.where(
    SCOTUS_df['docket'].str.contains('-'),
    SCOTUS_df['docket'],
    SCOTUS_df['docketId'].str[2:4] + '-' + SCOTUS_df['docket']
)

kaggle_selected_columns_df.loc[:, 'docket_standardized_kaggle'] = np.where(
    kaggle_selected_columns_df['docket'].str.contains('-'),
    kaggle_selected_columns_df['docket'],
    kaggle_selected_columns_df['term'].str[2:4] + '-' + kaggle_selected_columns_df['docket']
)

# Compute the merge
merged_df = SCOTUS_df.merge(kaggle_selected_columns_df, left_on='docket_standardized_SCOTUS', right_on='docket_standardized_kaggle', how='outer')


# In[7]:


# Consolidate merged dataframes
merged_df = merged_df.replace(np.nan, None)

merged_df['facts'] = np.where(merged_df['facts_x'], merged_df['facts_x'], merged_df['facts_y'])
merged_df['docket'] = np.where(merged_df['docket_standardized_SCOTUS'], merged_df['docket_standardized_SCOTUS'], merged_df['docket_standardized_kaggle'])
merged_df['term'] = np.where(merged_df['term_x'], merged_df['term_x'].astype(str), merged_df['term_y'].astype(str))
merged_df['first_party_winner'] = np.where(merged_df['partyWinning'], merged_df['partyWinning'].astype(bool), merged_df['first_party_winner'].astype(bool))

# Choose necessary columns from merged dataframe
df = merged_df[['docket', 'term', 'facts', 'first_party_winner']].copy()
print(df['first_party_winner'].dtype)

# Drop any row which does not have an outcome
df.dropna(subset=['first_party_winner'], inplace=True)

print(df)


# In[8]:


# Get year from the term value column 
def standardize_term(term):
    term = term[:4]
    return int(term)

df.loc[:, 'term'] = df['term'].apply(standardize_term)


# In[9]:


# NLP!!! 

import warnings

warnings.filterwarnings("ignore", module="urllib3")

import nltk
from nltk.corpus import stopwords
import regex as re
import spacy

# remove tags
df.loc[:, 'facts'] = df['facts'].apply(lambda s: re.sub(r'<[^>]+>', '', s))

# format text
df.loc[:, 'facts'] = df['facts'].apply(lambda s: re.sub(r'[^\w\s]', '', s.lower()))

# remove stopwords
# nltk.download('stopwords')
stopwords_english = set(stopwords.words('english'))
df.loc[:, 'facts'] = df['facts'].apply(lambda s: ' '.join([word for word in s.split(' ') if word not in stopwords_english])) 

# tokenizer
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

# lemmatize
df.loc[:, 'facts'] = [' '.join([word.lemma_ for word in facts]) for facts in nlp.pipe(df['facts'], batch_size=5000)]

print(df['facts'])


# In[10]:


# Given a year, determines a named court-era
def determine_era(year):
    year = int(year)
    if not year or year < 1942:
        return 'unk'
    elif 1942 <= year < 1986:
        return 'pre-rehnquist'
    elif 1986 <= year:
        return 'post-rehnquist_inc'

df.loc[:, 'era'] = df['term'].apply(determine_era)


# In[11]:


# Change first party winner to categorial variable for model training
df.loc[:, 'first_party_winner'] = df['first_party_winner'].astype('category')

# Save final full data
df.to_csv('./data/fully_processed_data')

# Filter data for non-recent and recent cases
df_pre_2021 = df[df['term'] < 2021]
df_post_2021_inc = df[df['term'] >= 2021]

# Save date-separated data
df_pre_2021.to_csv('./data/fully_processed_data_pre_2021')
df_post_2021_inc.to_csv('./data/fully_processed_data_post_2021_inc')


# In[12]:


# check sizes of each era's data
era_sizes = df_pre_2021['era'].value_counts().sort_index()
print(era_sizes)

# split into era
df_by_era = df_pre_2021.groupby(['era']).agg(list).reset_index()
df_by_era

# Save split-by-era data
df_by_era.to_csv('./data/data_by_era_pre_2021')


# Machine Learning

# In[13]:


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV 

# Given an era and column name, gets the series corresponding to that columns values for the era
def get_column_value_by_era(era, column):
    return df.loc[df['era'] == era, column]

# Given an era, returns the X_train, X_test, y_train, y_test for that era’s cases for use in model training
def train_test_split_for_era(era):
    X = get_column_value_by_era(era, 'facts')
    y = get_column_value_by_era(era, 'first_party_winner')
    
    return train_test_split(X, y, test_size=0.2,random_state=4120)
    
# Given a court era, trains a model on data from that era and scores the model on accuracy
def fit_and_score_model_for_era(era):
    X_train, X_test, y_train, y_test = train_test_split_for_era(era)

    param_grid = [
        {'classifier': [RandomForestClassifier()], 'classifier__n_estimators': [10, 100], 'classifier__max_features': [1, 2]},
        {'classifier': [LogisticRegression(max_iter=1000)], 'classifier__penalty': ['l2'], 'classifier__C': [1, 10]},
        {'classifier': [KNeighborsClassifier()], 'classifier__n_neighbors': [3, 5, 7], 'classifier__weights': ['uniform', 'distance']},
        # {'classifier': [XGBClassifier()], 'classifier__n_estimators': [100, 200], 'classifier__learning_rate': [0.01, 0.1], 'classifier__max_depth': [3, 6]}
    ]

    pipe = Pipeline(steps=[('cv', CountVectorizer()),
                            ('classifier', KNeighborsClassifier())])
                            
    grid_search = GridSearchCV(pipe, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_score_


# In[14]:


# Fit and score both models
pre_rehnquist_model, pre_rehnquist_model_score = fit_and_score_model_for_era('pre-rehnquist')
post_rehnquist_model, post_rehnquist_model_score = fit_and_score_model_for_era('post-rehnquist_inc')

# Show Model Scores
print('Pre-rehnquist-model score: ', pre_rehnquist_model_score)
print('Post-rehnquist-inclusive-model score: ', post_rehnquist_model_score)


# In[15]:


# Download models
from joblib import dump

dump(pre_rehnquist_model, './saved_models/pre_rehnquist_model')
dump(post_rehnquist_model, './saved_models/post_rehnquist_inc_model')

