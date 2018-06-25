# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 19:26:33 2018

@author: akshita
"""
import json
from sklearn.externals import joblib
from sklearn import preprocessing
import operator
from tqdm import tqdm
from collections import defaultdict

def getModalAnswer(answers):
	candidates = {}
	for i in range(10):
		candidates[answers[i]['answer']] = 1

	for i in range(10):
		candidates[answers[i]['answer']] += 1

	return max(candidates.items(), key=operator.itemgetter(1))[0]

def selectFrequentAnswers(answers_train, maxAnswers):
	answer_fq= defaultdict(int)
	#build a dictionary of answers
	for answer in answers_train:
		answer_fq[answer] += 1

	sorted_fq = sorted(answer_fq.items(), key=operator.itemgetter(1), reverse=True)[0:maxAnswers]
	top_answers, top_fq = zip(*sorted_fq)
	new_answers_train=[]
	#only those answer which appear int he top 1K are used for training
	for answer in answers_train:
		if answer in top_answers:
			new_answers_train.append(answer)
	return (new_answers_train)


b = 'E:/akshita_workspace/Audio-Vision/VIS-LSTM/data/annotations'
quesFile = b + '/v2_OpenEnded_mscoco_train2014_questions.json'
annFile  = b + '/v2_mscoco_train2014_annotations.json'
answers_file = open(b+'/preprocessed/answers_train2014_modal.txt', 'w')

max_answers=1000

questions = json.load(open(quesFile, 'r'))
ques = questions['questions']
qa = json.load(open(annFile, 'r'))
qa = qa['annotations']

for i in tqdm(range(len(ques))):
     answers_file.write(getModalAnswer(qa[i]['answers']))
     answers_file.write('\n')


answers_train = open(b+'/preprocessed/answers_train2014_modal.txt', 'r').read().splitlines()
new_answers_train=selectFrequentAnswers(answers_train,max_answers)
labelencoder = preprocessing.LabelEncoder()
labelencoder.fit(new_answers_train)
if len(labelencoder.classes_) != max_answers:
    raise Exception("Something went wrong")
joblib.dump(labelencoder,b+'/preprocessed/labelencoder_trainval.pkl')