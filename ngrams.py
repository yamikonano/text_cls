import csv
import pandas as pd
import numpy as np
import collections
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix, roc_curve, \
    precision_recall_curve, plot_roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter, namedtuple
import re
import unicodedata
import nltk
import seaborn as sns
# import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score, ShuffleSplit, cross_val_predict
from sklearn.pipeline import Pipeline
import json
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve, roc_auc_score
import scikitplot as skplt
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle
# import nltk
# import ssl
from nltk.util import ngrams
import warnings
import torch

# import utils as ul

# If there's a GPU Integable...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) Available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU Available, using the CPU instead.')
    device = torch.device("cpu")

warnings.filterwarnings("ignore")

# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


def text_preprocessing(text):
    # Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    text = text.lower()

    # Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
    text_words_list = word_tokenize(text)

    # Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(text_words_list):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
    return str(Final_words)


def basic_clean(text):
    """
    A simple function to clean up the data. All the words that
    are not designated as a stop word is then lemmatized after
    encoding and basic regex parsing are performed.
    """
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    text = (unicodedata.normalize('NFKD', text)
            .encode('ascii', 'ignore')
            .decode('utf-8', 'ignore')
            .lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]


np.random.seed(500)
d1 = pd.read_csv('input_Avail.csv', header=None, error_bad_lines=False, encoding="utf-8")
d1 = d1.drop(columns=[2, 3, 4, 5])
type = ['DESC', 'Avail']
d1 = d1.rename(columns={0: 'DESC', 1: 'Avail'})
d1 = d1.iloc[1:]
with open('rawFile.csv', 'w', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(type)
    writer.writerows(zip(d1['DESC'], d1['Avail']))

values = {'Partial': 0, 'Complete': 0, 'None': 0}
fsrc = open('rawFile.csv', 'r', encoding="utf-8")
fdst = open('tmpfile.csv', 'w', encoding="utf-8")

print(d1)

firstLine = True
for line in fsrc.readlines():
    if firstLine == True:
        fdst.write(line)
        firstLine = False
        continue
    tmpLine = line.strip("\n")
    tmpLine = tmpLine.strip(',')
    for riskLevelKey in values.keys():
        if tmpLine.endswith(riskLevelKey):
            if values[riskLevelKey] < 3000:
                fdst.write(line)
                values[riskLevelKey] = values[riskLevelKey] + 1
fsrc.close()
fdst.close()
print(values)

df = pd.read_csv('tmpfile.csv', encoding="utf-8")
df['Avail'].dropna(inplace=True)
words = basic_clean(''.join(str(df['DESC'].tolist())))
print(words)
df['final'] = df['DESC'].astype(str).map(text_preprocessing)

# bigrams_series = (pd.Series(nltk.ngrams(words, 2)).value_counts())[:12]
# trigrams_series = (pd.Series(nltk.ngrams(words, 3)).value_counts())[:12]

# bigrams_series.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))
# plt.title('20 Most Frequently Occuring Bigrams')
# plt.show()
# tfidf_vect, tfidf_vect_ngram, tfidf_vect_ngram_chars, vectorizer = ul.get_vectorize_ngrams()

# Generate n-grams upto n=1
vectorizer_ng1 = CountVectorizer(ngram_range=(1, 1))
ng1 = vectorizer_ng1.fit_transform(df['final'])

# Generate n-grams upto n=2
vectorizer_ng2 = CountVectorizer(ngram_range=(1, 2))
ng2 = vectorizer_ng2.fit_transform(df['final'])

# Generate n-grams upto n=3
vectorizer_ng3 = CountVectorizer(ngram_range=(1, 3))
ng3 = vectorizer_ng3.fit_transform(df['final'])

vectorizer_ng4 = CountVectorizer(ngram_range=(1, 4))
ng4 = vectorizer_ng4.fit_transform(df['final'])

print("ng1, ng2 and ng3 have %i, %i and %i features respectively" %
      (ng1.shape[1], ng2.shape[1], ng3.shape[1]))
print(ng2, ng1, ng3)
X = df['final']
y = df['Avail']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

ng_vectorizer = CountVectorizer(ngram_range=(1, 2))
X_train_ng = ng_vectorizer.fit_transform(X_train)
X_test_ng = ng_vectorizer.transform(X_test)




acc = []
pre = []
rec = []
f1c = []
Naive = naive_bayes.MultinomialNB()
SVM = svm.SVC(C=1, kernel='linear', probability=True)
rfc = RandomForestClassifier(max_depth=2, random_state=0)
dtc = DecisionTreeClassifier(random_state=0)
lgc = LogisticRegression(solver='liblinear',random_state=0)

for model in [Naive, SVM, rfc, dtc, lgc]:
    model.fit(X_train_ng, y_train)
    y_pred_class = model.predict(X_test_ng)
    print(model, accuracy_score(y_test, y_pred_class))
    print(model, precision_score(y_test, y_pred_class, average='weighted'))
    print(model, recall_score(y_test, y_pred_class, average='weighted'))
    print(model, f1_score(y_test, y_pred_class, average='weighted'))
