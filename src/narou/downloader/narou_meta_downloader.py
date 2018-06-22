import json
from urllib.request import urlopen
import gzip

def num_of_novels():
    url = 'http://api.syosetu.com/novelapi/api/?out=json&gzip=5&of=t&lim=1'
    response = urlopen(url)
    with gzip.open(response, 'rt', encoding='utf-8') as f:
        j_raw = f.read()
        j_obj = json.loads(j_raw)
    return j_obj[0]['allcount']

def fetch(st, limit):
    url = "http://api.syosetu.com/novelapi/api/?out=json&of=t-n-u-w-s-bg-g-k-nt-l-gp-f-r-a-ka&gzip=5&st={}&lim={}".format(st, limit)
    response = urlopen(url)
    with gzip.open(response, 'rt', encoding='utf-8') as f:
        j_raw = f.read()
    return json.loads(j_raw)

