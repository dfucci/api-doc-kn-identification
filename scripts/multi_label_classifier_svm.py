
# coding: utf-8
# In[73]:
import csv
import string

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

import nltk
from nltk.corpus import stopwords


# In[74]:
np.random.seed(0)
# In[75]:

print("Loading data")
test_df = pd.read_csv("test2.csv")
test_df = test_df[test_df.columns.drop(list(test_df.filter(regex='Unnamed*')))]
test_df = test_df[pd.notnull(test_df['text'])]


# In[76]:


train_df = pd.read_csv("train2.csv")
train_df = train_df[train_df.columns.drop(list(train_df.filter(regex='Unnamed*')))]
train_df = train_df[pd.notnull(train_df['text'])]


# #### Over- undersampling functions from Ali (see api-doc-knowledge-extraction/scripts/dataset_utils.py)
# - Undersample:
#     - _functionality_
#     - _non-information_
# - Oversample:
#     - _concept_
#     - _quality_
#     - _environment_

# # Note: Doing over- under-sampling only on the training data

# In[8]:


def load_data(data_path, header=True):
    data = open(data_path, 'r')
    data_reader = csv.reader(data, delimiter=',')
    if header:
        next(data_reader, None)  # skip the headers

    data = list(data_reader)
    return data


# In[9]:


def undersample_dataset(dataset, dims, n_samples):
    for dim in dims:
        col = np.array([d[dim] for d in dataset], dtype="int32")
        col_indexes = np.nonzero(col)
        random_indexes = np.random.choice(
            [x for xs in col_indexes for x in xs], n_samples, replace=False)

        elements = [dataset[x] for x in random_indexes]
        col_indexes = np.where(col < 1)
        col_indexes = [x for xs in col_indexes for x in xs]
        dataset_tmp = [dataset[x] for x in col_indexes]

        dataset = elements + dataset_tmp

    return dataset


# In[10]:

def oversample_dataset(dataset, dims, n_samples):

    for dim in dims:
        col = np.array([d[dim] for d in dataset], dtype="int")
        col_indexes = [x for xs in np.nonzero(col) for x in xs]
        random_indexes = np.random.choice(col_indexes, n_samples, replace=True)

        elements = [dataset[x] for x in random_indexes]
        col_indexes = np.where(col < 1)
        col_indexes = [x for xs in col_indexes for x in xs]
        dataset_tmp = [dataset[x] for x in col_indexes]

        dataset = elements + dataset_tmp

    return dataset


# In[11]:


# need to do this to conform with oversample_dataset and undersample_dataset functions from Ali
# train.to_csv('train_cado.csv')
# df_train = load_data('train_cado.csv')


# In[12]:


# df_train = undersample_dataset(df_train, [7, 18], 200)
# df_train = oversample_dataset(df_train, [8, 11, 16], 500)


# In[13]:


# add back the header. For some reason an additional column is added, remove it.
# df_train = pd.DataFrame(df_train, columns=train.columns.insert(item='extraID', loc=0))
# df_train = df_train.drop('extraID', axis=1)


# ## Creating cbow features (unigram and bigram as suggested by Nicole)

# In[77]:


# print(train_df.shape)
# print(test_df.shape)


# In[78]:


def create_bow_df(data, text_column):
    bow_transformer_uni = CountVectorizer(
        strip_accents='unicode', stop_words='english', ngram_range=(1, 1), max_features=800)

    bow_transformer_bi = CountVectorizer(
        strip_accents='unicode', stop_words='english', ngram_range=(1, 2), max_features=800)

    bow_transformer_uni.fit(data[text_column])
    bow_transformer_bi.fit(data[text_column])

    text_bow_uni = bow_transformer_uni.transform(data[text_column])
    text_bow_bi = bow_transformer_bi.transform(data[text_column])

    text_bow_uni_df = pd.DataFrame(text_bow_uni.toarray())
    text_bow_bi_df = pd.DataFrame(text_bow_bi.toarray())

    text_bow_df = pd.concat([text_bow_bi_df, text_bow_uni_df], axis=1)
    text_bow_df.columns = ['bow_' + str(col) for col in text_bow_df.columns]

    df = pd.concat([data, text_bow_df], axis=1)
    return df.dropna()


# In[79]:

print("creating bow")
df_train = create_bow_df(train_df, 'text')
df_test = create_bow_df(test_df, 'text')


# In[17]:


# df_train.to_csv('train.csv')
# df_test.to_csv('test.csv')


# In[80]:


clf = SVC()


# In[82]:


labels = ['functionality', 'concept', 'directives', 'purpose', 'quality', 'control',
    'structure', 'patterns', 'codeExamples', 'environment', 'reference', 'nonInformation']
y_test = df_test[labels]
y_train = df_train[labels]


# In[83]:


X_test = df_test.drop(labels, axis=1).filter(regex=("bow_*"))
X_train = df_train.drop(labels, axis=1).filter(regex=("bow_*"))


# # Training using SVM and GridSearch (Menzies suggests SVM + DE)

# In[85]:


scoring = ['precision_macro', 'recall_macro', 'f1_macro',
    'roc_auc', 'hamming_loss', 'accuracy_score']
# estimator__ necessary to get the SVC inside the OneVsRestClassifier
tuned_parameters = [{'estimator__kernel': ['rbf'], 'estimator__gamma': ['auto'],
                     'estimator__C': [0.01, 0.5, 1.0]},
                    {'estimator__gamma': ['auto'], 'estimator__kernel': ['linear'], 'estimator__C': [0.01, 0.5, 1.0]}]


# In[86]:


clf = SVC(class_weight="balanced", probability=True)
clf = OneVsRestClassifier(clf)
clf = GridSearchCV(clf, tuned_parameters, cv=10, verbose=10, n_jobs=4)


# In[87]:


def to_label(row):
    a = (list(map(int, row.values)))  # apparently Series.value returns str
    b = np.nonzero(a)
    arr = y_train.columns.values[b]
    return arr


# In[88]:

print('Creating labels')
y_train_labels = y_train.apply(to_label, axis=1)
y_test_labels = y_test.apply(to_label, axis=1)


# In[89]:


mlb = MultiLabelBinarizer()
y_train_enc = mlb.fit_transform(y_train_labels)
y_test_enc = mlb.fit_transform(y_test_labels)


# In[ ]:

print('Training classifiers')
clf.fit(X_train, y_train_enc)

print('Saving to file')
joblib.dump(clf, 'svm.pkl')
print('Calculating score')

for score in scoring:
    print(score),
    print(":"),
    print(cross_val_score(clf, X_test, y_test_enc, cv=10, n_jobs=4, verbose=10, scoring=score))

predictions = cross_val_predict(clf, X_test, y_test_enc, cv=10, n_jobs=4, verbose=10)
my_metrics= metrics.classification_report(y_test_enc, predictions)

#print(scores)
print (my_metrics)


# In[83]:

