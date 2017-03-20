
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
print("building inv-index")
bighash={}
inv_ind_hash={}
term_feq={}
doc_freq={}
file_tokens=[]
file_count=0
#root_dir= input("link of root_folder in " " without '/' in-the-end\n")

#import os
#print(os.path.dirname(os.path.abspath(__file__)))


import os.path
root_dir=os.path.abspath(os.path.join(os.getcwd(), os.pardir))
words_input_dir= root_dir + "/corpus"
#print(os.getcwd())
#print("hoo")
#words_input_dir=("")
#print(words_input_dir)
#words_input_dir="/Users/raghav/Desktop/untitledfolder"
for filename in os.listdir(words_input_dir):
	file_count=file_count+1
	temp={}
	if filename.endswith(".txt"):
		filename = os.path.join(words_input_dir, filename)
		with open(filename, "r") as input_file:
			file_tokens = nltk.word_tokenize(input_file.read().lower().decode('utf8', 'ignore'))
			for indx,word in enumerate(file_tokens):
				if st.stem(word) in temp.keys():
					temp[st.stem(word)].append(indx)
				else:
					temp[st.stem(word)]=[indx]
	bighash[filename]=temp
	
#pprint(bighash)		
#-INVERTED INDEX CONSTRUCTION STEPS GO HERE-------------------------
for curr_file in bighash.keys():
	for curr_word in bighash[curr_file].keys():
		if curr_word in inv_ind_hash.keys():
			if curr_file in inv_ind_hash[curr_word].keys():
				inv_ind_hash[curr_word][curr_file].extend(bighash[curr_file][curr_word][:])
			else:
				inv_ind_hash[curr_word][curr_file]=bighash[curr_file][curr_word]
		else:
			inv_ind_hash[curr_word] = {curr_file: bighash[curr_file][curr_word]}
				
				

print("building permuterm-index")
	
def rotate(str, n):
	return str[n:] + str[:n]
	
pindex={}
keys = inv_ind_hash.keys()
for key in sorted(keys):
	dkey = key + "$"
	for i in range(len(dkey),0,-1):
		out = rotate(dkey,i)
		pindex[out]=st.stem(key)
			
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
vectors={}
print("making document vectors for use")
for each_doc in bighash.keys():
	vectors[each_doc]=(make_doc_a_vect(each_doc))
	

	
#pickling to save running time by storing indexes in encoded form

print("pickling files")


file_Name = root_dir+"/pickled_files/pickled.txt"
fileObject = open(file_Name,'wb')
pickle.dump(inv_ind_hash,fileObject)
fileObject.close()

file_Name = root_dir+"/pickled_files/pickled2.txt"
fileObject = open(file_Name,'wb')
pickle.dump(bighash,fileObject)
fileObject.close()

file_Name = root_dir+"/pickled_files/pickled3.txt"
fileObject = open(file_Name,'wb')
pickle.dump(file_count,fileObject)
fileObject.close()

file_Name = root_dir+"/pickled_files/pickled4.txt"
fileObject = open(file_Name,'wb')
pickle.dump(pindex,fileObject)
fileObject.close()

file_Name = root_dir+"/pickled_files/pickled5"
fileObject = open(file_Name,'wb')
pickle.dump(vectors,fileObject)
fileObject.close()
print("-----------done indexing----------")

