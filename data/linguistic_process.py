from textblob import TextBlob
import numpy as np

	
#JJ - adj 0
#NN - noun
#RB - adv 1
#VB - verb
#PR - pron	2

etiq = { 'JJ': 0, 'RB': 1, 'PR': 2 }

def sentiment_analisis_textblob(texts,sa=True,sf=True):
	'''
	sa: sentiment analysis
	td: topic detection
	sf: syntactic features
	'''
	arr_text = []
	for text in texts:
		
		blob = TextBlob(text)
		arr = []		

		if sf:
			arr_tags = np.zeros(3)
			for tag in blob.tags:
				tag_add = tag[1][:2]
				if tag_add in etiq: arr_tags[etiq[tag_add]] += 1
			arr = arr_tags
				
		if sa:
			arr_sa = []
			
			for sentence in blob.sentences:
					arr_sa.append(sentence.sentiment.polarity)
			arr = np.mean(arr_sa)
		arr_text.append(arr)
				
	return arr_text
			
				
		
				
			
