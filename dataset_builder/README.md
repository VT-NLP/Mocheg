Dataset and crawlers for "Multi-modal fact-checking and explanation generation" project
 



# This is a crawler to generate the Corpus

## The repository contains or constructs the following corpora:

**Corpus 2:** Contains the information cralwed from the Snopes fact-checking websites: Claim, verdict, Evidence Text Snippets (ETSs), Resolution, ...

**Corpus 3:** Contains the the information about the links from the fact-checking websites and the original documents (Origin Docs) to which the links are pointing. 


The repository for the construction of the Corpora has two parts.
The first part is the crawler based on a Maven project, which is crawles the information (such as claim, evidence, ratings, Origin) of Snopes fact-checking pages 
from the web archives Wayback Machine and Common Crawl to generate the Corpus 2 and Corpus 3.
The second part is a python program for multimedia information.

In order to generate the annotated Corpus 2, which can be used for stance detection, evidence extraction and claim validation, run the following command:
	
	sudo apt install maven
	chmod +x script.sh
	./build_dataset.sh 
  
## Generating the  Corpus 2  

This command will generate Corpus2.csv in final_corpus directory. It contains the following important information:

**Claim:** The statement need to be verfied.

**Evidence:** It is a small text snippet in the blockquote of a Snopes fact-check website. It is a snippet that was extracted from an online article which may be related to the claim.

**Origin:** It is the article which provides the resolution of the claim.

**cleaned_truthfulness:** THe Rating it the given verdict to the claim.

**Snopes URL:** Each url is corresponding to an unique claim. It is where the information in the corpus that was extracted from.

**Commoncrawl URL:** This is where the website snapshot is stored by the commoncrawl.



## Generating the  Corpus 3  

This previous command will also generate Corpus3.csv in final_corpus directory. It contains the following important information:

**Claim_id:** The statement need to be verfied.

**Origin Doc:** The extracted text from a link in the Origin section of a Snopes fact-check website.

This corpus can be used for document retrieval. Given a claim, extract the related documents. 
 
 
 
