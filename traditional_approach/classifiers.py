from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectPercentile, chi2


#from sklearn.cross_validation import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

import numpy as np
from nltk import tokenize
import collections

from data import process

#dev = True
dev = False
min_ngram = 1
max_ngram = 1

def evaluate_dev(texts_train, authors_train, text_test, authors_test, clf, rep):

	texts_rep = rep.fit_transform(texts_train)

	#self.selector.fit(texts_rep,self.authors)
	#texts_rep = self.selector.transform(texts_rep).toarray()

	texts_rep = texts_rep.toarray()

	clf.fit(texts_rep, authors_train)

	text_test_rep = rep.transform(text_test)
	text_test_rep = text_test_rep.toarray()

	pred = clf.predict(text_test_rep) 

	f1 = f1_score(pred, authors_test, average='macro')
	acc = accuracy_score(pred, authors_test)
	return f1, acc

def evaluate_StratifiedKFold(texts_train, authors_train, clf, rep, k = 3):
	
	skf = StratifiedKFold(k)
	skf.get_n_splits(texts_train, authors_train)
	
	f1 = []
	acc = []
	texts = np.array(texts_train)
	y = np.array(authors_train)
	for train_index, test_index in skf.split(texts_train, authors_train):
		
		x_train, x_test = texts[train_index], texts[test_index]
		y_train, y_test = y[train_index], y[test_index]
		texts_rep = rep.fit_transform(x_train)
		
		texts_rep = texts_rep.toarray()
		text_test_rep = rep.transform(x_test)
		text_test_rep = text_test_rep.toarray()
		
		clf.fit(texts_rep, y_train)	
		pred = clf.predict(text_test_rep)
		f1.append(f1_score(pred, y_test, average='macro'))
		acc.append(accuracy_score(pred, y_test))
	#return np.mean(f1), np.mean(acc)
	return f1, acc

root = '/data2/jose/data_autoria/'
root = '../texts/'
train = root+"train"
test = root+"test"
vocab = root+"vocabulario"

dt_train = process.canon60Dataset(train)
dt_test = process.canon60Dataset(test)
vocab =  process.read_vocab(vocab)

x_train = dt_train.X
y_train = dt_train.y

x_test = dt_test.X
y_test = dt_test.y

del dt_train
del dt_test

'''
classifyer = {
		KNeighborsClassifier():'KNN',
		GaussianNB():'GNB',
		MultinomialNB():'MNB',
		LogisticRegression():'LR',
		DecisionTreeClassifier():'DTC',
		RandomForestClassifier():'RFC',
		SVC():'SVC',
             	MLPClassifier(hidden_layer_sizes=(32, 16)): "MLP",
		MLPClassifier(hidden_layer_sizes=(64, 32 )):"MLP2",
		MLPClassifier(hidden_layer_sizes=(128, 64, 32 )):"MLP3",
		MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32)):"MLP4",
		MLPClassifier(hidden_layer_sizes=(512,256,128, 64,32)):"MLP5"
}
'''

classifyer = {MLPClassifier(hidden_layer_sizes=(512,256,128, 64,32)):"MLP5"}

for clf in classifyer:
	for up in [1,2,3]:
		for voc in [True,False]:
			print(classifyer[clf])	
			print(up)
			print(voc)

			if voc: vocabulary = vocab
			else: vocabulary = None
			rep = TfidfVectorizer(ngram_range=(min_ngram,up),vocabulary=vocabulary)				
				
			if dev:	
				try:
					print(evaluate_dev(x_train, y_train, x_test, y_test, clf, rep))
				except:
					print('Dev pass')
			else:
				try:
					print(evaluate_StratifiedKFold(x_train, y_train, clf, rep))
				except:
					print('StratifiedKFold pass')
			


'''
#x_train = ' '.join([x.replace("\n", " ") for x in x_train])
#x_test = ' '.join([x.replace("\n", " ") for x in x_test])

#n = 2 # character n-gram
#x_train_char_grams = ' '.join([i for i in ["".join(j) for j in zip(*[x_train[i:] for i in range(n)])] if i.isalpha()])

from nltk import ngrams

n = 2 # word n-gram
ngramas = ngrams(text.split(), n)
x_train_word_grams = ' '.join(['_'.join(x_train) for gram in ngramas])


author_dicc = collections.Counter(y_train)  # authors dictionary - {author: number of texts}
author_list = [aut for aut in author_dicc]  # authors list
index = author_list.index(aut)              # index given an author
aut = author_list[index]                    # author given an index

'''




