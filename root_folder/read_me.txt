READ ME


1) download the root folder
2) copy the link to root folder, and input with " ", when asked
3)if you want to change the corpus, copy all the new .txt files to root_folder/corpus
4)run root_folder/code_files/indexing.py first to build and store the index
5)run root_folder/code_files/querying.py to query.


NOTE:
1) Wild card queries have to be only of the form "query*" "*query"
2) phrase queries have to be exactly what you want in the document
3)corpus can be dynamic in the sense that add/delete new txt file to the corpus directory and execute indexing.py again.





EXPLANATION:
pickled files contain the indices for faster retrieval of query results,
it contains 
1)posting lists 
2)permuterm_indices for wild card queries
3)
