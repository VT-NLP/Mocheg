import pandas as pd
import csv
# def add_header(twitter_dir):
#     file = pd.read_csv(twitter_dir+"/twitter_data_new.csv")   
#     # adding header
#     headerList = ['ID', 'Claim_id', 'Relevant_doc_id', 'snoopes_url', 'link_url' 'Text', 'Media_Type', 'Media URL']
    
#     # converting data frame to csv
#     file.to_csv("twitter_data_new.csv", header=headerList, index=False)
import shutil 
def merge(parent_dir,twitter_dir):
    import pandas as pd
    data_df = pd.read_csv(twitter_dir+"/corpus_3_twitter_data.csv")
    
        
    for _,row in data_df.iterrows():
            
        claim_id = row['claim_id']
        relevant_doc_id = row['relevant_document_id']
        snopes_url = row['Snopes URL']
        link_url = row['Link URL']
        archive_url=row["Archive Url"]
        text = row['Origin Document']

        data = [claim_id, relevant_doc_id,  snopes_url, link_url, archive_url, text] 

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
    
    

def split_image_for_one_split(data_path,splited_data_path)    :
    image_corpus=os.path.join(data_path ,"images") 
    img_names=os.listdir(image_corpus)
    i=0
    length=len(img_names)
    for filepath in  img_names:
         
        source_path=os.path.join(image_corpus,filepath)
        target_path=os.path.join(os.path.join(splited_data_path ,"images") ,filepath)
        os.rename(source_path,target_path )
          
    print("finish split_image_for_one_split")
    
import os 
def merge_main(data_dir):
    twitter_dir=data_dir+ "/twitter"
    merge(data_dir,twitter_dir)
    split_image_for_one_split(twitter_dir,data_dir)
    
    
# main()
