import csv
import cPickle

import pandas as pd
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from textblob import TextBlob


__author__ = 'laurogama'

# messages = [line.rstrip() for line in open('../datasets/sms_spam/SMSSpamCollection')]
# print len(messages)
# for message_no, message in enumerate(messages[:10]):
# print message_no, message
def split_into_tokens(message):
    message = unicode(message, 'utf8')  # convert bytes into proper unicode
    return TextBlob(message).words


def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]


def execute_tfid(message):
    bow = bow_transformer.transform([message])
    return tfidf_transformer.transform(bow)

# print messages
# print messages.groupby('label').describe()
# print messages.message.head().apply(split_into_tokens)
# print messages.message.head().apply(split_into_lemmas)

messages = pd.read_csv('../datasets/sms_spam/SMSSpamCollection', sep='\t', quoting=csv.QUOTE_NONE,
                       names=["label", "message"])
# message4 = messages['message'][3]
# print message4
bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
messages_bow = bow_transformer.transform(messages['message'])
# print 'sparse matrix shape:', messages_bow.shape
# print 'number of non-zeros:', messages_bow.nnz
#
# print 'sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)

# print messages_tfidf.shape

spam_detector = MultinomialNB().fit(messages_tfidf, messages['label'])

# print 'predicted:', spam_detector.predict(execute_tfid(messages['message'][3]))[0]
# print 'expected:', messages.label[3]
all_predictions = spam_detector.predict(messages_tfidf)
# print all_predictions
# print classification_report(messages['label'], all_predictions)

msg_train, msg_test, label_train, label_test = \
    train_test_split(messages['message'], messages['label'], test_size=0.2)

# print len(msg_train), len(msg_test), len(msg_train) + len(msg_test)
pipeline_svm = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),  # <== change here
])

# pipeline parameters to automatically explore and tune
param_svm = [
    {'classifier__C': [1], 'classifier__kernel': ['linear']},
    # {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
]

grid_svm = GridSearchCV(
    pipeline_svm,  # pipeline from above
    param_grid=param_svm,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)
svm_detector = grid_svm.fit(msg_train, label_train)  # find the best combination from param_svm

print svm_detector.predict(["Hi mom, how are you?"])[0]
print svm_detector.predict(["WINNER! Credit for free!"])[0]


def store_classifier(svm_detector):
    # store the spam detector to disk after training
    with open('sms_spam_detector.pkl', 'wb') as fout:
        cPickle.dump(svm_detector, fout)


def reload_classifier(filename):
    # 'sms_spam_detector.pkl'
    # ...and load it back, whenever needed, possibly on a different machine
    return cPickle.load(open(filename))