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

class classify():
	'''
	Classifying with sklearn
	'''

	def __init__(self, texts, classes, classifier = SVC(), rep = TfidfVectorizer(), k=3, sel = SelectPercentile(chi2, percentile=10)):
		self.texts = texts
		self.features = None
		self.authors = classes
		self.classifier = classifier
		self.name = 'SVC'
		self.representation = rep
		self.selector = sel
		self.K = k

	def set_clf(self,clf,name=''):
		self.classifier = clf
		self.name = name

	def set_rep(self,rep):
		self.representation = rep

	def training_classify(self):
		texts_rep = self.representation.fit_transform(self.texts)

		#self.selector.fit(texts_rep,self.authors)
		#texts_rep = self.selector.transform(texts_rep).toarray()

		texts_rep = texts_rep.toarray()

		self.classifier.fit(texts_rep, self.authors)
		return texts_rep

	def classify(self,text_test):
		text_test_rep = self.representation.transform(text_test)

		#text_test_rep = self.selector.transform(text_test_rep).toarray()

		text_test_rep = text_test_rep.toarray()
	
		pred = self.classifier.predict(text_test_rep)   
		return pred

	def evaluate_dev(self,text_test, authors_test):
		texts_rep = self.training_classify()
		text_test_rep = self.representation.transform(text_test)
		text_test_rep = text_test_rep.toarray()
		pred = self.classifier.predict(text_test_rep) 
		f1 = f1_score(pred, authors_test, average='macro')
		acc = accuracy_score(pred, authors_test)
		return f1, acc
		

	def evaluate_StratifiedKFold(self,clf=SVC(),rep=TfidfVectorizer(),feature=False):

		skf = StratifiedKFold(n_splits=self.K)
		skf.get_n_splits(self.texts, self.authors)

		f1 = []
		acc = []
		texts = np.array(self.texts)
		y = np.array(self.authors)
		for train_index, test_index in skf.split(self.texts, self.authors):
			
			x_train, x_test = texts[train_index], texts[test_index]
			y_train, y_test = y[train_index], y[test_index]

			rep_temp = rep
			clf_temp = clf
			texts_rep = rep_temp.fit_transform(x_train)

			texts_rep = texts_rep.toarray()
			text_test_rep = rep_temp.transform(x_test)
			text_test_rep = text_test_rep.toarray()
			
			clf_temp.fit(texts_rep, y_train)	
			pred = clf_temp.predict(text_test_rep)
			f1.append(f1_score(pred, y_test, average='macro'))
			acc.append(accuracy_score(pred, y_test))
		return np.mean(f1), np.mean(acc)

	def evaluate(self,clf=SVC()):
		texts_rep = self.training_classify()
		kf = KFold(self.K)
		scores = cross_val_score(clf, texts_rep, self.authors, cv=kf)
		avg_score = np.mean(scores)
		return avg_score


	def evaluate2(self,text_test, authors_test):
		results = cross_validate(lasso, self.texts, self.authors, cv=self.K,return_train_score=True)
		
		# The score for test 
		test_score_avg = np.mean(results['test_score']) 

		# The score for train scores 
		train_score_avg = np.mean(results['train_score'])

		# The time for fitting the estimator on the train set 
		fit_time_avg = np.mean(results['fit_time'])

		# The time for scoring the estimator on the test set 
		score_time_avg = np.mean(results['score_time'])

	
	def get_features(self):
		texts_features = []
		for x in self.texts:
			x_vector = []
			sentences = tokenize.sent_tokenize(x)
			#number_words_avg = np.mean([len(s.split()) for s in sentences])
			size = []
			punct = []
			for s in sentences:
				size.append(len(s.split()))
				punct.append(len([char for char in s if char in string.punctuation]))
			x_vector.append(np.mean(size)) #average number of words per sentence
			x_vector.append(np.mean(punct)) #average number of punctuation per sentence

			texts_features.append(x_vector)
		self.features = texts_features



root = '/data2/jose/data_autoria/'
train = root+"train"
test = root+"test"
vocab = root+"vocabulario"

dt_train = process.canon60Dataset(train)
dt_test = process.canon60Dataset(test)

x_train = dt_train.X
y_train = dt_train.y

x_test = dt_test.X
y_test = dt_test.y

del dt_train
del dt_test

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

Classify = classify(x_train, y_train)
#print(Classify.evaluate()) 

del x_train
del y_train

classifyer = {
	MLPClassifier(hidden_layer_sizes=(32, 16)): "MLP",
	MLPClassifier(hidden_layer_sizes=(64, 32 )):"MLP2",
	MLPClassifier(hidden_layer_sizes=(128, 64, 32 )):"MLP3",
	MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32)):"MLP4",
	MLPClassifier(hidden_layer_sizes=(512,256,128, 64,32)):"MLP5",
}
# classifyer = {KNeighborsClassifier():'KNN',GaussianNB():'GNB',MultinomialNB():'MNB',LogisticRegression():'LR',DecisionTreeClassifier():'DTC',RandomForestClassifier():'RFC',SVC():'SVC'}

'''
TF-IDF-based representation 
'''
for clf in classifyer:
	Classify.set_clf(clf,classifyer[clf])
	print(Classify.name)	
	#print(Classify.evaluate_dev(x_test,y_test))
	print(Classify.evaluate_StratifiedKFold(clf))
	#print(Classify.evaluate(clf))


'''
Linguistic Features-based representation 

Classify.get_features()
for clf in classifyer:
	Classify.set_clf(clf,classifyer[clf])
	print(Classify.name)
	print(Classify.evaluate(clf))
	#print(Classify.evaluate_StratifiedKFold(clf,feature=True))
'''


#Results

## evaluate_dev: (f1, acc)

# KNN: (0.895963849160704, 0.9111842105263158)

# GBN: (0.43864583080816394, 0.6387061403508771)

# MNB: (0.09213211269397649, 0.4791666666666667)

# LR: (0.7434306828103954, 0.825109649122807)

# DTC: (0.7592064248304642, 0.774671052631579)

# RFC: (0.6618323427589164, 0.7214912280701754)

# SVC: (0.01748271724847169, 0.25164473684210525)


## evaluate_StratifiedKFold(K=3): (f1, acc)

# KNN: (0.881304754584138, 0.8986532057579506)

# GBN: (0.41073519247857004, 0.6216048052443551)

# MNB: (0.08389227989802306, 0.4705112985362337)

# LR: (0.6745793295194847, 0.7946699695595872)

# DTC: (0.7063509760317418, 0.7243485134738497)

# RFC: (0.6456953499639223, 0.7060202919335627)

# SVC: (0.017550477210018272, 0.25286699163670995)



