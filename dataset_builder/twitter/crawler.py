import pandas as pd
import tweepy
import webbrowser
import os 
import urllib.request
from urllib.error import URLError, HTTPError

def get_twitter_url(LinkCorpus_file):
    tweet_urls = []
    claim_ids = []
    relevant_doc_ids = []
    snopes_url = []
    count = 0
    data = pd.read_csv(LinkCorpus_file)
    urls = data['Original Link URL']
    ids = data['claim_id']
    r_ids = data['relevant_document_id']
    snops = data['Snopes URL']
    for index, url in enumerate(urls):
        words = url.split('/')
        if 'www.twitter.com' in words:
            tweet_urls.append(url)
            claim_ids.append(ids[index])
            relevant_doc_ids.append(r_ids[index])
            snopes_url.append(snops[index])
            count+=1
        elif 'twitter.com' in words:
            tweet_urls.append(url)
            claim_ids.append(ids[index])
            relevant_doc_ids.append(r_ids[index])
            snopes_url.append(snops[index])
            count+=1
    print(count)
    return tweet_urls,claim_ids,relevant_doc_ids,snopes_url

def get_status_id(tweet_urls,claim_ids,relevant_doc_ids,snopes_url):
    from urllib.parse import urlparse
    import re
    status_id = []
    new_claim_ids = []
    new_rel_doc_ids = []
    new_snoops = []
    new_tweet_urls = []

    for idx, item in enumerate(tweet_urls):
        url = item
        df = urlparse(url).path.split('/')
        for item in df:
            if len(item) ==19 or len(item) == 18:
                status_id.append(item)
                new_claim_ids.append(claim_ids[idx])
                new_rel_doc_ids.append(relevant_doc_ids[idx])
                new_snoops.append(snopes_url[idx])
                new_tweet_urls.append(tweet_urls[idx])
    print(f"{len(status_id)}, {len(new_claim_ids)}")
    return  status_id,new_tweet_urls,new_claim_ids,new_rel_doc_ids,new_snoops
    
    
def get_auth():
    consumer_key = "kSsK1G38xcyMHnhXOXyurxrOH"
    consumer_secret = "SiVcbtucDekrZtlBndFkNRsxj5p2AF6hli3rrabd3Shced7BYd"
    callback_uri = 'oob'
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret, callback_uri)
    redirect_url = auth.get_authorization_url()
    # webbrowser.open(redirect_url)
    # user_pin_input = input('whats the pin value ')
    # auth.get_access_token(user_pin_input)
     
    api = tweepy.API(auth)
    return api
 
 
def fetch_tweets(parent_dir,api,status_id,new_tweet_urls,new_claim_ids,new_rel_doc_ids,new_snoops):
    import pickle
    import time

    tweet_data = []
    claim_ids = []
    rel_doc_ids = []
    snoopes_url = []
    twitter_url = [] 
     
    for idx, id in enumerate(status_id):
        
        try:
            data = api.get_status(id , tweet_mode='extended')._json
            tweet_data.append(data)
            claim_ids.append(new_claim_ids[idx])
            rel_doc_ids.append(new_rel_doc_ids[idx])
            snoopes_url.append(new_snoops[idx])
            twitter_url.append(new_tweet_urls[idx])
            parse_one_tweet(parent_dir,data,new_claim_ids[idx],new_rel_doc_ids[idx],new_snoops[idx],new_tweet_urls[idx],idx)
            time.sleep(1)
        except Exception as e: 
            print(e)
            print(id)
            print('-'*20)
            time.sleep(1)
        
     
        if idx %100==0:
            print(idx)
        
    with open(os.path.join(parent_dir,'tweets.pickle'), 'wb') as handle:
        pickle.dump(tweet_data, handle)
        
    return claim_ids,rel_doc_ids,snoopes_url,twitter_url


def stop_handled( cursor):
    while True:
        try:
            yield cursor.next()
        except StopIteration:
            return 
import csv
def write_csv(parent_dir,data):
    with open(os.path.join(parent_dir,'twitter_data_new.csv'), 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)

import pickle
import json

import pandas as pd
def parse_one_tweet(parent_dir,tweet,claim_id,relevant_doc_id,snoop,tweet_url,idx):
    image_dir = parent_dir+'/images'
    videos_dir  =parent_dir+'/videos'
 
    text = tweet['full_text']
    id = tweet['id_str']
    prefix=str(claim_id).zfill(5)+"-"+str(relevant_doc_id).zfill(5)+"-"
    photo_url = []
    video_url = []
    media_type = None
    photo_count = 0
    video_count = 0

    if 'extended_entities' in tweet:
        for item in tweet['extended_entities']['media']:
            media_type = item['type']
            
            if media_type == 'photo':
                url = item['media_url_https']
                path = os.path.join(image_dir,prefix+ str(photo_count) +"-"+ id)   + ".jpg"
                photo_count+=1
                try:
                    urllib.request.urlretrieve(url,path)
                except HTTPError as e:
                    print('&'*20)
                    print(url)
                    print('&'*20)
                photo_url.append(url)

            elif media_type == 'video':
                variants = item['video_info']['variants']

                for data in variants:

                    if data['content_type'] == 'video/mp4':

                        if data["bitrate"] == 832000 or data["bitrate"] == 632000:

                            url = data['url']
                            path = os.path.join(videos_dir,prefix+ str(video_count) +"-"+ id)   + ".mp4"

                            video_count+=1
                    
                            try:
                                urllib.request.urlretrieve(url,path)
                                
                            except HTTPError as e:
                            # do something
                                print('&'*20)
                                print(url)
                                print('&'*20)


                            video_url.append(url)
                            continue
                        
                        

    if media_type == None:
        dat = [str(id), claim_id, relevant_doc_id, snoop, tweet_url, text, 'None', 'None']
        write_csv(parent_dir,dat)

    elif media_type == 'photo':
        dat = [str(id), claim_id, relevant_doc_id, snoop, tweet_url, text, media_type, photo_url]
        write_csv(parent_dir,dat) 

    elif media_type ==  'video':
        dat = [str(id), claim_id, relevant_doc_id, snoop, tweet_url, text, media_type, video_url]
        write_csv(parent_dir,dat) 

                

    if idx<5:
        print("Text:", text)
        print("CLAIM ID:", claim_id)
        print("REL DOC ID", relevant_doc_id)
        print("Media type:", media_type)
        print("Photo_url:", photo_url )
        print("Video url:", video_url)
        print(dat)
        print('-'*100)




def main():
    LinkCorpus_file = 'final_corpus/politifact_v1/LinkCorpus.csv'
    parent_dir="final_corpus/politifact_v1/twitter"
    tweet_urls,claim_ids,relevant_doc_ids,snopes_url=get_twitter_url(LinkCorpus_file)
    status_id,new_tweet_urls,new_claim_ids,new_rel_doc_ids,new_snoops=get_status_id(tweet_urls,claim_ids,relevant_doc_ids,snopes_url)
    api=get_auth()
    claim_ids,rel_doc_ids,snoopes_url,twitter_url=fetch_tweets(parent_dir,api,status_id,new_tweet_urls,new_claim_ids,new_rel_doc_ids,new_snoops)
    # parse_tweet(parent_dir,claim_ids,rel_doc_ids,snoopes_url,twitter_url)
    
    
    
main()
    

 