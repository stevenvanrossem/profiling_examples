import os
import yaml
import logging
import requests
from random import randint, random, choice
import itertools

logging.basicConfig(filename='fill_cache.log',level=logging.DEBUG)

HTTP_PROXY = "10.10.0.1:3128"
os.environ["http_proxy"] = HTTP_PROXY

def readConfig(path):
    yml = None
    with open(path, "r") as f:
            try:
                yml = yaml.load(f)
            except yaml.YAMLError as ex:
                logging.exception("YAML error while reading %r." % path)
    return yml

def genFileSizes(min=1, max=10, scaler=1):
    filesizes = list(range(min,max+1))
    new_filesizes = [ (x / scaler) for x in filesizes]
    return new_filesizes

def genFileIds(numFiles=10):
    return list(range(1,numFiles+1))

CONFIG = readConfig("client_config.yml")
config = CONFIG['cached']
min = config['filesize']['min']
max = config['filesize']['max']
scaler = config['filesize']['scaler']
filesizes = genFileSizes(min=min, max=max, scaler=scaler)
fileids = genFileIds(config['objects'])


HTTP_PROXY = "10.10.0.1:3128"
os.environ["http_proxy"] = HTTP_PROXY


# iterate over all files to fill the cache first
for file in itertools.product(fileids, filesizes):
    url = "http://10.20.0.2:8888/file/{}/{}".format(file[0], file[1])
    r = requests.get(url)
    print(url)
    print(r)
    logging.info("{} {}".format(url, r))

