import pandas as pd
import csv
# def add_header(twitter_dir):
#     file = pd.read_csv(twitter_dir+"/twitter_data_new.csv")   
#     # adding header
#     headerList = ['ID', 'Claim_id', 'Relevant_doc_id', 'snoopes_url', 'link_url' 'Text', 'Media_Type', 'Media URL']
    
#     # converting data frame to csv
#     file.to_csv("twitter_data_new.csv", header=headerList, index=False)
    
def merge(parent_dir,twitter_dir):
    import pandas as pd
    data_df = pd.read_csv(twitter_dir+"/twitter_data_new.csv")
    
        
    for _,row in data_df.iterrows():
            
        claim_id = row['Claim_id']
        relevant_doc_id = row['Relevant_doc_id']
        snopes_url = row['snoopes_url']
        link_url = row['link_url']
        text = row['Text']

        data = [claim_id, relevant_doc_id,  snopes_url, link_url, " ", text] 

        write_csv(parent_dir,data)
            
            
def write_csv(parent_dir,data):
    with open(parent_dir+'/Corpus3.csv', 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)



 
def statistic(parent_dir,twitter_dir):
    import pandas as pd
    data_df = pd.read_csv(twitter_dir+"/twitter_data_new.csv")
    print(len(data_df))
    return len(data_df)
    
    


def main():
    parent_dir="final_corpus/politifact_v2"
    twitter_dir=parent_dir+ "/twitter"
  
    # merge(parent_dir,twitter_dir)
    snopes_sum=statistic("/home/aditya/misinformation/data/twitter_data","/home/aditya/misinformation/data/twitter_data")
    politifact_sum=statistic(parent_dir,twitter_dir)
    print(snopes_sum+politifact_sum)
    
    
main()
