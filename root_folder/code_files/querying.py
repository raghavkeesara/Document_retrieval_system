import pickle
import nltk
import re
from nltk.stem import SnowballStemmer
import numpy as np
import math
import codecs
import difflib
import os, sys
from pprint import pprint
st = SnowballStemmer('english')
#stopwords_filename = "words.txt"
#with open(stopwords_filename, "r") as stopwords_file:
	#stop_words = stopwords_file.read()
	
root_dir=os.path.abspath(os.path.join(os.getcwd(), os.pardir))
words_input_dir= root_dir + "/corpus"
file_Name = root_dir +"/pickled_files/pickled.txt"
fileObject = open(file_Name,'r')  
inv_ind_hash = pickle.load(fileObject) 


file_Name = root_dir +"/pickled_files/pickled5.txt"
fileObject = open(file_Name,'r')  
vectors = pickle.load(fileObject) 


file_Name = root_dir +"/pickled_files/pickled2.txt"
fileObject = open(file_Name,'r')  
bighash = pickle.load(fileObject) 

file_Name = root_dir +"/pickled_files/pickled3.txt"
fileObject = open(file_Name,'r')  
file_count = pickle.load(fileObject) 

file_Name = root_dir +"/pickled_files/pickled4.txt"
fileObject = open(file_Name,'r')  
pindex = pickle.load(fileObject) 

#-all freuency measure calc functions go here------------------------
def tf(doc,word):
	if word in inv_ind_hash.keys():
		if doc in inv_ind_hash[word].keys():
			return len(inv_ind_hash[word][doc])
		else:
			return 0
	else:
		return 0


def doc_freq(word):
	if word in inv_ind_hash.keys():
		return len(inv_ind_hash[word].keys()) 
	else:
		return 0
		
def idf(word):
	if word in inv_ind_hash.keys():
		if doc_freq(word)!=0:

			return (math.log((file_count-1)/float(doc_freq(word))));
		else:
			return 0
	else:
		return 0

	
def tf_idf(word,doc):
		return tf(doc,word)*idf(word)

#- VECTORIZATION OF DOCUMENTS---------------------------------------
def make_doc_a_vect(curr_doc):
		doc_vect={}
		magn=0
		for curr_word in inv_ind_hash.keys():
			doc_vect[curr_word]=0;	
		for each_word in bighash[curr_doc].keys():
			doc_vect[each_word]=tf_idf(each_word, curr_doc)
		for each_word in inv_ind_hash.keys():
			magn=magn+doc_vect[each_word]*doc_vect[each_word]
		for each_word in inv_ind_hash.keys():
			if magn!=0:
				doc_vect[each_word]=doc_vect[each_word]/math.sqrt(magn)
		
		return doc_vect	
		
#all-query-processing goes here--------------------------------------


def single_word_unRanked(query_input):
	print(difflib.get_close_matches(st.stem(query_input), inv_ind_hash)[0])
	if st.stem(query_input) in inv_ind_hash.keys():
		return inv_ind_hash[st.stem(query_input)].keys()
	if difflib.get_close_matches(st.stem(query_input), inv_ind_hash)[0] in inv_ind_hash.keys():
		return inv_ind_hash[difflib.get_close_matches(st.stem(query_input), inv_ind_hash)[0]].keys()
	else:
		return []

def many_word_unRanked(query_input):
	pattern = re.compile('[\W_]+')
	query_input = pattern.sub(' ',query_input)
	result = []
	for word in query_input.split():
		result += single_word_unRanked(word)
		
	return result
	
	
def tf_of_query(check_term, query_input):
	count=0;
	for query_word in query_input.split():
		if query_word == check_term:
			count=count+1
	return count;
	
def rankResults(resultDocs, query_input):
	vectors={}
	for each_res in resultDocs:
		vectors[each_res]=(make_doc_a_vect(each_res))
	print("-------------------------")
	queryVec = make_query_a_vect(query_input)
	results = [[dotProduct(vectors[result], queryVec), result] for result in resultDocs]
	results.sort(key=lambda x: x[0])
	results = [x[1] for x in results]
	return results	
	
def make_query_a_vect(query_input):
	query_vect={}
	query_tokens = nltk.word_tokenize(query_input.lower())
	for i in range(0,len(query_tokens)-1):
		query_tokens[i]=st.stem(query_tokens[i])
	for curr_word in inv_ind_hash.keys():
		query_vect[curr_word]=0;
	for curr_word in query_tokens:
		if curr_word in inv_ind_hash.keys():
			query_vect[curr_word]=tf_of_query(curr_word,query_input)*idf(curr_word)
	magn=0
	for each_word in inv_ind_hash.keys():
		magn=magn+query_vect[each_word]*query_vect[each_word]
	for each_word in inv_ind_hash.keys():
		if magn!=0:
			query_vect[each_word]=query_vect[each_word]/math.sqrt(magn)
	return 	query_vect

	
def get_score(doc):
	score=0
	for curr_word in inv_ind_hash.keys():
		score=score+tf_idf(curr_word, doc)	
	return score

			
def dotProduct(doc1, doc2):
	dotpro=0
	if len(doc1) != len(doc2):
		return "error"
	else:
		for each_word in inv_ind_hash.keys():
			dotpro += doc1[each_word]*doc2[each_word]
	return dotpro
	
def rankedResults(query_input):
	queryVec = make_query_a_vect(query_input)
	results = [[dotProduct(vectors[result], queryVec), result] for result in bighash.keys()]
	#pprint(results)
	results.sort(key=lambda x: x[0])
	results.reverse()
	pprint(results)
	results = [x[1] for x in results]
	return results
	
def spellcheck(query_input):	
	tokenized_query = nltk.word_tokenize(query_input)
	normalized_query = [st.stem(word.lower()) for word in tokenized_query]
	spl_cor_query = []
	for i in range(0,len(normalized_query)):
		spl_cor_query.append(difflib.get_close_matches(normalized_query[i], inv_ind_hash)[0])
	return spl_cor_query

def phrase_query(query_input):
	pattern = re.compile('[\W_]+')
	query_input = pattern.sub(' ',query_input)
	listOfLists, result = [],[]
	for word in query_input.split():
		listOfLists.append(single_word_unRanked(word))
	setted = set(listOfLists[0]).intersection(*listOfLists)
	for filename in setted:
		temp = []
		for word in query_input.split():
			temp.append(inv_ind_hash[st.stem(word)][filename][:])
		for i in range(len(temp)):
			for ind in range(len(temp[i])):
				temp[i][ind] -= i
		if set(temp[0]).intersection(*temp):
			result.append(filename)
	return rankResults(result, query_input)
	
def prefix_match(pindex, prefix):
	term_list = []
	for tk in pindex.keys():
		if tk.startswith(prefix):
			term_list.append(st.stem(pindex[tk]))
	return term_list
	
def find_wc(query_input):
	parts = query_input.split("*")
	if len(parts) == 3:
		case = 4
	elif parts[1] == '':
		case = 1
	elif parts[0] == '':
		case = 2
	elif parts[0] != '' and parts[1] != '':
		case = 3
		
	if case == 1:
		query_shortened= parts[0]
	elif case == 2:
		query_shortened = parts[1] + "$"
	elif case == 3:
		query_shortened = parts[1] + "$" + parts[0]
	elif case == 4:
		queryA = parts[2] + "$" + parts[0]
		queryB = parts[1]
	if case != 4:
		process_wc_Query(query_shortened)
		
		
def process_wc_Query(query_shortened):
	term_list = prefix_match(pindex,query_shortened)
	for t in term_list:
		print("t is ")
		pprint(t)
		pprint(single_word_unRanked(t))
	return
	


def main():
	print("1= normalised query vector.")
	print("2= spell correction of the query.")
	print("3= documents with BR-model, ranked with tf-idf.")
	print("4= phrase query.")
	print("5= wc query.")
	print("6= results with ranked-tf-idf-model")
	print("7= exit")
	
	while(1):
		query_input= raw_input("------ask your query-------\n")
		user = input("press 1/2/3/4/6 or 7 to exit\n")
		if(user==6):
			rankedResults(query_input)
		if(user==7):
			break
		if(user==1):
			pprint(make_query_a_vect(query_input))
		if(user==2):
			print(spellcheck(query_input))
		if(user==4):
			pprint(phrase_query(query_input))
		if(user==5):
			find_wc(query_input)
		if(user==3):
			pprint(rankResults(many_word_unRanked(query_input), query_input))
		#if(user==6):
		#	parts = query_input.split("*")
			
main()
	
