import pandas as pd
import numpy as np
import re
import pickle
from nltk import ngrams
from gensim.models import Word2Vec,KeyedVectors
from sklearn.pipeline import Pipeline
# from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


df=pd.read_csv("Final_file.csv")
df.drop('Unnamed: 0',axis=1,inplace=True)
df.dropna(inplace=True)
df.reset_index(inplace=True,drop=True)
try_df=df

print("Lemmatization files loading...")
lemma_words=pd.read_csv("Lemma.csv")
lemma_words['input']=lemma_words['input'].str.split()

print("Train w2c formation....")
def n_grams(lst):
    bigram=["_".join(ph) for ph in list(ngrams(lst,2))]
#     trigram=["_".join(ph) for ph in list(ngrams(lst,3))]
    return bigram

train_w2v=[]
for i in lemma_words['input']:
    lst=i
    bi=n_grams(lst)
    train_w2v.append(lst + bi)

# abc=list(model.wv.index_to_key)

load_model=KeyedVectors.load('word2vec.model')
words = set(load_model.wv.index_to_key )

print("Model loaded and words set formed")
lbl=try_df['label'].iloc[:60000]
X1_train=train_w2v[:60000]
Y1_train=lbl

X_train, X_test, Y_train, Y_test = train_test_split(X1_train, Y1_train,test_size=0.3,random_state=0,shuffle=False)

print("train_vect")
train_vect = np.array( [np.array( [load_model.wv[word] for word in sent if word in words],dtype=object) for sent in X_train],dtype=object)
train_vect_avg = []
for v in train_vect:
    if v.size:
        train_vect_avg.append(v.mean(axis=0))
    else:
        train_vect_avg.append(np.zeros(50, dtype=float))
print("train_vect")
test_vect = np.array( [np.array( [load_model.wv[word] for word in sent if word in words],dtype=object) for sent in X_test[:10000]],dtype=object)
test_vect_avg = []
for v in test_vect:
    if v.size:
        test_vect_avg.append(v.mean(axis=0))
    else:
        test_vect_avg.append(np.zeros(50, dtype=float))

print("pipeline starts")
def func_text_data(x):
    return x

def func_numeric_data(x):
    return x

get_text_data = FunctionTransformer(func_text_data, validate=False)
get_numeric_data = FunctionTransformer(func_numeric_data, validate=False)

textual_pipeline=Pipeline([ ('selector', get_text_data) ])
numerical_pipeline=Pipeline([ ('selector', get_numeric_data) ])

combined_features = FeatureUnion([('numeric_features', numerical_pipeline), ('text_features', textual_pipeline)])

pipeline = Pipeline([
    ('features',combined_features),
    ('clf', RandomForestClassifier())])

param_grid = {'clf__n_estimators': np.linspace(1, 100, 10, dtype=int),
              'clf__min_samples_split': [3, 10],
              'clf__min_samples_leaf': [3],
              'clf__max_depth': [None],
              'clf__criterion': ['entropy'],
              'clf__bootstrap': [False]}

rf_model = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=-1, return_train_score=True, verbose=1,error_score='raise')
rf_model.fit(train_vect_avg,Y_train)

print("model saving")
with open('pipeline_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)