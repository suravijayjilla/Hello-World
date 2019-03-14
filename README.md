# Required Packages
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
              'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
              'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc',
              'talk.religion.misc']

data = fetch_20newsgroups(subset='train', categories=categories, random_state=42, shuffle=True)
len(data.data)
len(data.filenames)
"\n".join(data.data[5].split("\n")[:3])
print(data.target_names[data.target[5]])
print(data.target[:10])
# for t in data.target[:10]:
#     data.target_names[t]

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train_counts = cv.fit_transform(data.data)
#X_train_counts.shape
cv.vocabulary_.get(u'algorithm')

from sklearn.feature_extraction.text import TfidfTransformer
tft = TfidfTransformer()
X_train_tft = tft.fit_transform(X_train_counts)
#X_train_tft.shape

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
classifier = clf.fit(X_train_tft, data.target)
docs = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = cv.transform(docs)
X_new_tft = tft.transform(X_new_counts)
predicted = clf.predict(X_new_tft)

for doc, category in zip(docs, predicted):
    "%r => %s" %(doc, data.target_names[category])

from sklearn.pipeline import Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('nb', MultinomialNB()),
])
text_clf.fit(data.data, data.target)

import numpy as np
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted1 = text_clf.predict(docs_test)
np.mean(predicted1 == twenty_test.target)

from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                alpha=1e-3, random_state=42,
                                    max_iter=5, tol=None)),
])

text_clf.fit(data.data, data.target)
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)

from sklearn import metrics
metrics.classification_report(twenty_test.target, predicted,target_names=twenty_test.target_names)
metrics.confusion_matrix(twenty_test.target, predicted)

from sklearn.model_selection import GridSearchCV
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)
gs_clf = gs_clf.fit(data.data[:400], data.target[:400])
print(data.target_names[gs_clf.predict(['Hardware'])[0]])
print(gs_clf.best_score_)
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
