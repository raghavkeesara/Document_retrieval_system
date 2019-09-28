# Text based Search Engine
This is a project done by me, as a part of my Information Retrieval course's assignment(also attached descriptive sheet), user can ask queries and retrieve documents relevant to the query from the corpus
1) download the root folder
2)if you want to change the corpus, copy all the new .txt files to root_folder/corpus
4)run root_folder/code_files/indexing.py first to build and store the index
5)run root_folder/code_files/querying.py to query.


NOTE:
1) Wild card queries have to be only of the form "query*" "*query", need a to write a small code for queries of the form A*B(working on it)
2) phrase queries have to be exactly what you want in the document
3) corpus can be dynamic in the sense that add/delete new txt file to the corpus directory and execute indexing.py again.


EXPLANATION:
pickled files contain the indices for faster retrieval of query results,
it contains 
1)posting lists 
2)intermediate hashtable
3)file_count value
4)permuterm_indices for wild card queries
3)vectors
