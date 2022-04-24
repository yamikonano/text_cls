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
import warnings
import torch

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



if __name__ == "__main__":
    # try:
    #     _create_unverified_https_context = ssl._create_unverified_context
    # except AttributeError:
    #     pass
    # else:
    #     ssl._create_default_https_context = _create_unverified_https_context
    #
    # nltk.download()
    # Set Random seed
    np.random.seed(500)
    d1 = pd.read_csv('input_conf.csv',header=None,error_bad_lines=False,encoding="utf-8")
    d1 = d1.drop(columns=[2,3,4,5])
    type=['DESC','Conf']
    d1 = d1.rename(columns={0: 'DESC', 1: 'Conf'})
    d1 = d1.iloc[1:]
    with open('rawFile.csv','w',encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(type)
        writer.writerows(zip(d1['DESC'],d1['Conf']))

    values = {'Partial':0,'Complete':0,'None':0}
    fsrc = open('rawFile.csv','r',encoding="utf-8")
    fdst = open('tmpfile.csv','w',encoding="utf-8")

    print(d1)

    # print(type(tmpInputFile))

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
    # # tmpline = tmpline[:-1]
    #
    # # N-gram
    # # def ngram(documents, N=2):
    # #     ngram_prediction = dict()
    # #     total_grams = list()
    # #     words = list()
    # #     Word = namedtuple('Word', ['word', 'prob'])
    # #
    # #     for doc in documents:
    # #         split_words = ['<s>'] + list(doc) + ['</s>']
    # #         # 計算分子
    # #         [total_grams.append(tuple(split_words[i:i + N])) for i in range(len(split_words) - N + 1)]
    # #         # 計算分母
    # #         [words.append(tuple(split_words[i:i + N - 1])) for i in range(len(split_words) - N + 2)]
    # #
    # #     total_word_counter = Counter(total_grams)
    # #     word_counter = Counter(words)
    # #
    # #     for key in total_word_counter:
    # #         word = ''.join(key[:N - 1])
    # #         if word not in ngram_prediction:
    # #             ngram_prediction.update({word: set()})
    # #
    # #         next_word_prob = total_word_counter[key] / word_counter[key[:N - 1]]
    # #         w = Word(key[-1], '{:.3g}'.format(next_word_prob))
    # #         ngram_prediction[word].add(w)
    # #
    # #     return ngram_prediction

    # d1 = pd.read_csv('input_Integ.csv',error_bad_lines=False)
    # Step - 1: Data Pre-processing - This will help in getting better results through the classification algorithms

    # Step - 1a : Remove blank rows if any.
    print("Pre-process the text...")
    # print(d1['DESC'])
    df = pd.read_csv('tmpfile.csv',encoding="utf-8")
    df['Conf'].dropna(inplace=True)

    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    df['final'] = df['DESC'].astype(str).map(text_preprocessing)
    # d1['final'] = text_preprocessing(d1['DESC'])
    print(df['final'])

    df['final'] = df['final'].astype(str)
    df['Conf'] = df['Conf'].astype(str)
    # Step - 2: Split the model into Train and Test Data set
    # by sklearn library
    # training set 70%, test set 30%
    # x --> predictor, y --> target
    # Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(d1['final'], d1['label'],
    #                                                                 test_size=0.3,random_state=0)

    # Step - 3: Label encode the target variable  - This is done to transform Categorical data of string type in the data set into numerical values
    Encoder = LabelEncoder()
    Encoder.fit(df['Conf'].astype(str))
    # Train_Y_Raw = Train_Y
    Train_Y_Raw = Encoder.transform(df['Conf'].astype(str))
    # Test_Y = Encoder.tralEncoder()
    # Encoder.fit(Train_Y)
    # Train_Y_Raw = Train_Y
    # Train_Y = Encoder.transform(Train_Y)
    # Test_Y = Encoder.transform(Test_Y)

    # Step - 4: Vectorize the words by using TF-IDF Vectorizer - This is done to find how important a word in document is in comaprison to the corpus
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(df['final'])
    Train_X_Tfidf = Tfidf_vect.transform(df['final'])
    # Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    # Test_X_Tfidf = Tfidf_vect.transform(Test_X)

    # BoW
    # count_vect = CountVectorizer().fit(d1['final'])
    # Train_X_Tfidf = count_vect.fit_transform(Train_X)
    # Test_X_Tfidf = count_vect.transform(Test_X)

    # N-gram

    # pipe = Pipeline([('vec', CountVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=100000)),
    #                  ('clf', LogisticRegression(C=4, dual=True))])
    # vec = pipe.fit(d1['final'])
    # Train_X_Tfidf = vec.fit.transform(Train_X)
    # Test_X_Tfidf = vec.transform(Test_X)
    # clf = pipe.fit(d1['final'], d1['label'])

    print("\nTrain and test model...")
    # Step - 5: Now we can run different algorithms to classify out data check for accuracy
    # Classifier - Algorithm - Naive Bayes
    # fit the training dataset on the classifier
    acc = []
    pre = []
    rec = []
    f1c = []
    Naive = naive_bayes.MultinomialNB()
    SVM = svm.SVC(C=1, kernel='linear', probability=True)
    rfc = RandomForestClassifier(max_depth=2, random_state=0)
    dtc = DecisionTreeClassifier(random_state=0)
    lgc = LogisticRegression(random_state=0)
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    names = ['Naive', 'SVM', 'Random Forest', 'Decision Tree', 'Logistic Regression']
    sampling_methods = [Naive, SVM, rfc, dtc, lgc]
    colors = ['crimson',
              'orange',
              'gold',
              'mediumseagreen',
              'steelblue',
              'mediumpurple'
              ]
    for (name, method, colorname) in zip(names, sampling_methods, colors):
        for train_index, test_index in cv.split(Train_X_Tfidf, Train_Y_Raw):
            X_train, X_test = Train_X_Tfidf[train_index], Train_X_Tfidf[test_index]
            y_train, y_test = Train_Y_Raw[train_index], Train_Y_Raw[test_index]
            method.fit(X_train, y_train)
            y_test_preds = method.predict(X_test)
            # y_test_predprob = method.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, y_test_preds, pos_label=2, sample_weight=None,
                                             drop_intermediate=True)

        plt.plot(fpr, tpr, lw=2, label='{} (AUC={:.3f})'.format(name, metrics.auc(fpr, tpr)), color=colorname)
        plt.plot([0, 1], [0, 1], '--', lw=4, color='grey')
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate', fontsize=5)
        plt.ylabel('True Positive Rate', fontsize=5)
        plt.title('ROC Curve', fontsize=5)
        plt.legend(loc='lower right', fontsize=5)
    plt.show()
    # if save:
    #     plt.savefig('multi_models_roc.png')

    for model in [Naive, SVM, rfc, dtc, lgc]:
        for train_index, test_index in cv.split(Train_X_Tfidf, Train_Y_Raw):
            X_train, X_test = Train_X_Tfidf[train_index], Train_X_Tfidf[test_index]
            y_train, y_test = Train_Y_Raw[train_index], Train_Y_Raw[test_index]

            model.fit(X_train, y_train)
            y_pred_class = model.predict(X_test)
            acc.append(accuracy_score(y_test, y_pred_class))
            pre.append(precision_score(y_test, y_pred_class, average='weighted'))
            rec.append(recall_score(y_test, y_pred_class, average='weighted'))
            f1c.append(f1_score(y_test, y_pred_class, average='weighted'))

        print(model, ' accuracy score: ', acc)
        acc = []
        print(model, ' precision score: ', pre)
        pre = []
        print(model, ' recall score: ', rec)
        rec = []
        print(model, ' f-1 score: ', f1c)
        f1c = []
        #
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_class, pos_label=2, sample_weight=None, drop_intermediate=True)
        roc_auc = metrics.auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        cm = confusion_matrix(y_test, y_pred_class)
        print('Confusion matrix\n\n', cm)
        print('\nTrue Positives(TP) = ', cm[0, 0])
        print('\nTrue Negatives(TN) = ', cm[1, 1])
        print('\nFalse Positives(FP) = ', cm[0, 1])
        print('\nFalse Negatives(FN) = ', cm[1, 0])
        cm_matrix = pd.DataFrame(data=cm,
                                 columns=['Complete','Partial','None'],
                                 index=['Complete','Partial','None'])
        sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
        plt.show()
    # exit()
    # def matrix(model):
    #     acc = []
    #     pre = []
    #     rec = []
    #     f1c = []
    #     acc.append(accuracy_score(y_test, y_pred_class))
    #     pre.append(precision_score(y_test, y_pred_class, average='weighted'))
    #     rec.append(recall_score(y_test, y_pred_class, average='weighted'))
    #     f1c.append(f1_score(y_test, y_pred_class, average='weighted'))
    #     return acc,pre,rec,f1c

    #     cm=[]
    #     metrics.append(accuracy_score(y_test, y_pred_class))
    # metrics = np.array(metrics)
    # print('Mean accuracy: ', metrics)
    # print('Std for accuracy: ', np.std(metrics, axis=0))
    # for model in ['Naive', 'SVM']:
    #     for score in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']:
    #         scores = cross_val_score(model, Train_X_Tfidf, Train_Y_Raw, cv=cv, scoring=score)
    #         print(model, end=' ')
    #         print(score, end='')
    #         print(':', end='')
    #         print(scores)
    # clf.fit(Train_X_Tfidf, Train_Y_Raw)
    # Naive.fit(Train_X_Tfidf,Train_Y_Raw)
    # Pred_SVM = cross_val_predict(clf, Train_X_Tfidf, Train_Y_Raw, cv=5)
    # print(Pred_SVM.ndim)

    # print("\nTrain and test Naive bayes...")
    # Naive = naive_bayes.MultinomialNB()
    # Naive.fit(Train_X_Tfidf, Train_Y)
    # Use accuracy_score function to get the accuracy
    #     print("Naive Bayes Accuracy Score -> ", accuracy_score(Test_Y, predictions_NB) * 100)
    #     print("Naive Bayes Precision Score -> ", precision_score(Test_Y, predictions_NB, average='weighted') * 100)
    #     print("Naive Bayes Recall Score -> ", recall_score(Test_Y, predictions_NB, average='weighted') * 100)
    #     print("Naive Bayes F-1 Score -> ", f1_score(Test_Y, predictions_NB, average='weighted') * 100)
    #
    #     # Classifier - Algorithm - SVM
    #     # fit the training dataset on the classifier
    #     print("\nTrain and test SVM...")
    #     SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto',probability=True)
    #     SVM.fit(Train_X_Tfidf, Train_Y)
    #
    #     # Evaluating
    #     Pred_Y_SVM = SVM.predict(Test_X_Tfidf)
    #     print("SVM Accuracy Score -> ", accuracy_score(Test_Y, Pred_Y_SVM) * 100)
    #     print("SVM Precision Score -> ", precision_score(Test_Y, Pred_Y_SVM, average='weighted') * 100)
    #     print("SVM Recall Score -> ", recall_score(Test_Y, Pred_Y_SVM, average='weighted') * 100)
    #     print("SVM F-1 Score -> ", f1_score(Test_Y, Pred_Y_SVM, average='weighted') * 100)
    #
    #     # cm = precision_recall_curve(Test_Y,Pred_Y_SVM)
    #     # print(cm)
    #     # plt.imshow(cm, cmap='binary')
    #     # plt.show()
    #     # fig, ax = plt.subplots(figsize=(12, 10))
    #     # lr_roc = plot_roc_curve(SVM,Test_Y.reshape(-1, 1),Pred_Y_SVM, ax=ax, linewidth=1)
    #     # ax.legend(fontsize=12)
    #     # plt.show()
    #
    #     print("\nTrain and test random forest...")
    #     clf = RandomForestClassifier(max_depth=2, random_state=0)
    #     clf.fit(Train_X_Tfidf, Train_Y)
    #     Pred_Y_RandomForest = clf.predict(Test_X_Tfidf)
    #     print("Random forest Accuracy Score -> ", accuracy_score(Test_Y, Pred_Y_RandomForest) * 100)
    #     print("Random forest Precision Score -> ", precision_score(Test_Y, Pred_Y_RandomForest, average='weighted') * 100)
    #     print("Random forest Recall Score -> ", recall_score(Test_Y, Pred_Y_RandomForest, average='weighted') * 100)
    #     print("Random forest F-1 Score -> ", f1_score(Test_Y, Pred_Y_RandomForest, average='weighted') * 100)
    #
    #     print("\nTrain and test decision tree...")
    #     clf1 = DecisionTreeClassifier(random_state=0)
    #     clf1.fit(Train_X_Tfidf, Train_Y)
    #     Pred_Y_DecisionTree = clf1.predict(Test_X_Tfidf)
    #     print("Decision Tree Accuracy Score -> ", accuracy_score(Test_Y, Pred_Y_DecisionTree) * 100)
    #     print("Decision Tree Precision Score -> ", precision_score(Test_Y, Pred_Y_DecisionTree, average='weighted') * 100)
    #     print("Decision Tree Recall Score -> ", recall_score(Test_Y, Pred_Y_DecisionTree, average='weighted') * 100)
    #     print("Decision Tree F-1 Score -> ", f1_score(Test_Y, Pred_Y_DecisionTree, average='weighted') * 100)
    #
    #     print("\nTrain and test Logistic Regression...")
    #     clf2 = LogisticRegression(random_state=0).fit(Train_X_Tfidf, Train_Y)
    #     clf2.fit(Train_X_Tfidf, Train_Y)
    #     Pred_Y_LogisticRegression = clf2.predict(Test_X_Tfidf)
    #     print("Logistic Regression Accuracy Score -> ", accuracy_score(Test_Y, Pred_Y_LogisticRegression) * 100)
    #     print("Logistic Regression Precision Score -> ", precision_score(Test_Y, Pred_Y_LogisticRegression, average='weighted') * 100)
    #     print("Logistic Regression Recall Score -> ", recall_score(Test_Y, Pred_Y_LogisticRegression, average='weighted') * 100)
    #     print("Logistic Regression F-1 Score -> ", f1_score(Test_Y, Pred_Y_LogisticRegression, average='weighted') * 100)
    #
    #
    #     # Part 2: Generate one sentence describing the risk level of the vulnerability
    #     # Pre-defined template:
    #     #       The security risk level of the vulnerability is [predicted_level]
    #
    #     # Here we use the result of SVM classfiers as the example
    #     # SVM best performance: SVM F-1 Score ->  81.39690551216371
    #
    # print(X_test)
    #     # Step 1: get the index and the description of each vulnerability in the test set
    #     Test_X_index2DescriptionTexts = {}
    #     for index, value in X_test.items():
    #         Test_X_index2DescriptionTexts[str(index)] = value
    #     # Step 2: get the raw label of each vulnerability in the test set
    Test_Y_RawLabels = Encoder.inverse_transform(y_test)
    Test_Y_PredLabels = Encoder.inverse_transform(y_test_preds)  # Use the predicted SVM results

    #     # Step 3: Display the text and predicted result of each sample in test set
    #     #   Generate the summary for the vulnerability
    index = 0
    for test_X_text in X_test:
        a = Tfidf_vect.inverse_transform(test_X_text)
        print("\n>>>Test Sample:", a)
        test_Y_rawLabel = Test_Y_RawLabels[index]
        Predicted_Label = Test_Y_PredLabels[index]
        print("Raw label:", test_Y_rawLabel, "\tPredicted label:", Predicted_Label)
        GeneratedSentence = "The security risk level of the vulnerability is " + str(Predicted_Label)
        print("Summary: " + GeneratedSentence)
        index = index + 1
        if index > 10:  # Only describe the top 50 samples in the test set
            break

    #
    # # # predict the labels on validation dataset
    # # predictions_SVM = SVM.predict(Test_X_Tfidf)
    # #
    # # svm_prob = SVM.predict_proba(Train_X_Tfidf)
    # # # probability for positive outcome is kept
    # # svm_prob = svm_prob[:,1]
    # #
    # # svm_auc = roc_auc_score(Test_Y, svm_prob,multi_class='ovo')
    # # print("SVM: AUROC = %.3f" %(svm_prob))
    # # fpr, tpr, _ =roc_curve(Test_Y, svm_prob)
    # #
    # # # plot ROC curve
    # # plt.plot(fpr, tpr, marker='.', label = "SVM (AUROC %0.3f)" %(svm_auc))
    # # plt.title('ROC curve of SVM')
    # # plt.xlabel('False positive rate')
    # # plt.ylabel('True positive rate')
    # # plt.legend()
    # # plt.show()
    # # # Use accuracy_score function to get the accuracy
    # # print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y) * 100)
    #
    #
    # Saving Encdoer, TFIDF Vectorizer and the trained model for future infrerencing/prediction

    # saving encoder to disk
    # filename = 'labelencoder_fitted.pkl'
    # pickle.dump(Encoder, open(filename, 'wb'))
    # #
    # saving TFIDF Vectorizer to disk
    # filename = 'Tfidf_vect_fitted.pkl'
    # pickle.dump(Tfidf_vect, open(filename, 'wb'))

    # saving the both models to disk
    # filename = 'svm_trained_model_Integ.sav'
    # pickle.dump(SVM, open(filename, 'wb'))

    # # # filename = 'nb_trained_model.sav'
    # # # pickle.dump(Naive, open(filename, 'wb'))

    print("Files saved to disk! Proceed to inference.py")
exit()
