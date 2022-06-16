
from link_crawler.Image.Download import Download
import os
import argparse
import os
import pandas as pd
from urllib.parse import urlparse
import sys
from newspaper import Article
import urllib.request
import pandas as pd
import newspaper
from newspaper import Config
# import wget
user_agent_str="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36"#"Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"

# url_to_crawl="util/tried_image_crawler/test_url3.csv"
 
def fetch_img(args):
    data_path=args.data_path
    ORIGIN_LINK_CORPUS="LinkCorpus.csv"
    url_to_crawl=os.path.join(data_path,ORIGIN_LINK_CORPUS)
    out_dir=args.out_dir
    is_direct_dir=True
    resume_claim_id=0 
    df_evidence = pd.read_csv(url_to_crawl ,encoding="utf8" )
 
    init()
    cur_snope_url=""
    snope_id=-1
    run_dir=gen_run_dir(out_dir,is_direct_dir )
    for relevant_document_id,row in df_evidence.iterrows():
        snope_url=row["Snopes URL"]
        origin_doc_url=row["Original Link URL"]
        snope_id=row["claim_id"]
        relevant_document_id=row["relevant_document_id"]
        if snope_id>=resume_claim_id:  
            if snope_url!=cur_snope_url:
                cur_snope_url=snope_url
                try:
                    fetch_img_by_newspaper(snope_url, snope_id,"proof",run_dir)
                except Exception as e:
                    print(e)
            try:
                fetch_img_by_newspaper(origin_doc_url, snope_id,relevant_document_id,run_dir)
            except Exception as e:
                print(e)

def gen_run_dir(out_dir,is_direct_dir):
    if is_direct_dir:
        return out_dir 
    else:
        dir_list = next(os.walk(out_dir))[1]

        max_run_id=0
        for dir in dir_list:
            run_id=int(dir[len(dir)-3:])
            max_run_id=max(run_id,max_run_id)
        return out_dir+"/run"+"{:0>3d}".format(max_run_id+1)
 

def init():
    # Adding information about user agent
    opener=urllib.request.build_opener()
    opener.addheaders=[('User-Agent',user_agent_str)]
    urllib.request.install_opener(opener)
    
def fetch_img_by_newspaper(url, snope_id,evidence_id,run_dir):
    user_agent =user_agent_str
    config = Config()
    config.browser_user_agent = user_agent
    article = Article(url.strip(), config=config)
    
    article.download()

    article.parse()
    
 
    

    filtered_imgs=filter(article.images)
    prefix=str(snope_id).zfill(6)+"-"+str(evidence_id).zfill(6)
    download = Download(run_dir,prefix,links=filtered_imgs)
    download.start()


def filter(images):
    filter_flags=[".svg",".gif",".ico","lazyload",".cgi","logo","-ad-","Logo",".php","icon","Bubble","svg%","rating-false",
    "rating-true","banner","-line"]
    filtered_imgs=[]
    for img_url in  images:
        should_remove=False
        for filter_flag in filter_flags:
            if  filter_flag   in img_url   :
                should_remove=True
        if not should_remove:    
            filtered_imgs.append(img_url)
    return filtered_imgs


def parser_args():

    parser = argparse.ArgumentParser()
 
    parser.add_argument('--data_path',type=str,help=" ",default="../final_corpus/politifact_v1")
    parser.add_argument('--out_dir',type=str,help=" ",default="../final_corpus/politifact_v1")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parser_args()
    fetch_img(args)

