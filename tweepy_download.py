import tweepy

consumer_key="vKBUwDBPaqEBZZ86BABBlNCFb"
consumer_secret="bqnjqtKAdhtsHd6nAZj0cXWUbTeWL1TYvAuiFhZESfVyoXECBJ"

access_token="771594014-N4u8cajP8DTdeusllAqsFzgXOJLDd59V6v7MhP4Q"
access_token_secret="dYXesNmIYkpIjb7JpveqEJ2JfvSkzXBnMz0E3JT9xzlk2"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

partier = [
# "konservativedk",
# "venstredk",
# "alternativet_",
# "enhedslisten",
# "sfpolitik",
# "nyeborgerlige",
# "radikale",
# "Spolitik",
# "liberalalliance",
"danskdf1995"
]

for p in partier:
    print("doing " + p)
    counter = 0
    with open(p+'_unicode.json', 'w') as outf:
        for status in tweepy.Cursor(api.user_timeline, screen_name=p, tweet_mode="extended").items():
            counter = counter + 1
            if counter % 500 == 0:
                print(counter)
            outf.write(json.dumps(status._json, ensure_ascii=False))
            outf.write("\n")
            
import json
import os

infiles = [f for f in os.listdir() if 'unicode' in f]


for f in infiles:
    with open(f) as inf:
        handle =  f.split("_")[0]
        with open(handle+'_out.json', 'w') as outf:
            for line in inf.readlines():
                counter  = counter + 1
                outdict = {}
                indict = json.loads(line)
                outdict['twitter_handle'] = handle
                outdict['time'] = indict['created_at']
                outdict['id'] = indict['id']
                outdict['id_str'] = indict['id_str']
                if 'retweeted_status' in  indict:
                    outdict['full_text'] = indict['retweeted_status']['full_text']
                else:
                    outdict['full_text'] = indict['full_text']
                outf.write(json.dumps(outdict, ensure_ascii=False))
                outf.write("\n")
            
