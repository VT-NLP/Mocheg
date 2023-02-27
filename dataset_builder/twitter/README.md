  
 
To obtain our tweet corpus and merge it into the public dataset:
1, You can find twitter_id.csv in the current dataset
2, put the twitter_id.csv under the "data" folder

```3, run
PYTHONPATH=. python  dataset_builder/twitter/crawler.py --consumer_key=#Your_twitter_consumer_key --consumer_secret=#Your_twitter_consumer_secret
```
 