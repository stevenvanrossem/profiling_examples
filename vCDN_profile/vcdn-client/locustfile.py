#!/usr/bin/env python

from prometheus_client import start_http_server, Summary, Histogram, Gauge, Counter, REGISTRY, CollectorRegistry, \
    pushadd_to_gateway, push_to_gateway, delete_from_gateway
from subprocess import Popen, PIPE, STDOUT
import threading, queue
from time import sleep
import pty
import os
import re
import sys
from math import isnan
import requests
import time
import signal
from random import randint, random, choice

from locust import HttpLocust, TaskSet, events, task

import yaml

import logging
logging.basicConfig(format='%(name)s:%(levelname)s %(module)s:%(lineno)d:  %(message)s', level="DEBUG")

VNF_NAME = os.environ.get('VNF_NAME', 'test')
#pushgateway = 'localhost:9091'
PUSHGATEWAY = '172.17.0.1:9091'

# Prometheus export data
# helper variables to calculate the metrics
VCDN_REGISTRY = CollectorRegistry()

PROM_FILESIZE = Summary('filesize', 'requested file sizes',
                       ['vnf_name'], registry=VCDN_REGISTRY).labels(vnf_name=VNF_NAME)

PROM_PROCESSED_CACHED_REQS = Counter('processed_cached_reqs', 'cached requests',
                         ['vnf_name'], registry=VCDN_REGISTRY).labels(vnf_name=VNF_NAME)
PROM_PROCESSED_NON_CACHED_REQS = Counter('processed_non_cached_reqs', 'cached requests',
                               ['vnf_name'], registry=VCDN_REGISTRY).labels(vnf_name=VNF_NAME)

PROM_FALSE_USERS = Counter('false_users', 'false requests',
                         ['vnf_name'], registry=VCDN_REGISTRY).labels(vnf_name=VNF_NAME)
PROM_CACHED_USERS = Counter('cached_users', 'cached requests',
                         ['vnf_name'], registry=VCDN_REGISTRY).labels(vnf_name=VNF_NAME)
PROM_NON_CACHED_USERS = Counter('non_cached_users', 'cached requests',
                               ['vnf_name'], registry=VCDN_REGISTRY).labels(vnf_name=VNF_NAME)

PROM_PROCESSED_FALSE_REQS = Counter('processed_false_reqs', 'false requests',
                         ['vnf_name'], registry=VCDN_REGISTRY).labels(vnf_name=VNF_NAME)

PROM_INPUT_CACHED_REQS = Counter('input_cached_reqs', 'cached requests',
                         ['vnf_name'], registry=VCDN_REGISTRY).labels(vnf_name=VNF_NAME)
PROM_INPUT_NON_CACHED_REQS = Counter('input_non_cached_reqs', 'cached requests',
                               ['vnf_name'], registry=VCDN_REGISTRY).labels(vnf_name=VNF_NAME)

PROM_CACHED_REQS_CURRENT = Gauge('cached_reqs_count', 'number of ongoing cached requests',
                             ['vnf_name'], registry=VCDN_REGISTRY).labels(vnf_name=VNF_NAME)
PROM_NON_CACHED_REQS_CURRENT = Gauge('non_cached_reqs_count', 'number of ongoing non-cached requests',
                             ['vnf_name'], registry=VCDN_REGISTRY).labels(vnf_name=VNF_NAME)

PROM_CACHED_SPEED = Summary('cached_download_speed', 'cached download speed',
                       ['vnf_name'], registry=VCDN_REGISTRY).labels(vnf_name=VNF_NAME)
PROM_NON_CACHED_SPEED = Summary('non_cached_download_speed', 'non-cached download speed',
                            ['vnf_name'], registry=VCDN_REGISTRY).labels(vnf_name=VNF_NAME)

#start_http_server(8000)

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


CONFIG = readConfig("client_config.yml")


def export_metrics():
    while True:
        # push metrics to gateway
        pushadd_to_gateway(PUSHGATEWAY, job='vcdn_client', registry=VCDN_REGISTRY)
        sleep(1)


export_thread = threading.Thread(target=export_metrics)
export_thread.start()


def genFileSize(min=1, max=10, scaler=1):
    return float(randint(min, max))/scaler


def genFileId(numFiles=10):
    return randint(1, numFiles)


def genBool():
    return random() < 0.5


class RandomGetter(TaskSet):

    @task(1)
    def browse(self):
        # random cached or not cached
        cached = genBool()

        if not cached:
            headers = {'Cache-Control': 'no-cache'}
            PROM_NON_CACHED_REQS_CURRENT.inc()
        else:
            headers = {}
            PROM_CACHED_REQS_CURRENT.inc()

        # random file size
        size = choice([0.5, 5, 50])
        #size = genFileSize(1, 5)
        url = "/file/{}".format(size)

        start = time.time()
        r = self.client.get(url, headers=headers)
        total_time = (time.time() - start)

        total_length = int(r.headers.get('content-length'))
        PROM_FILESIZE.observe(total_length)
        total_speed = total_length // total_time
        if not cached:
            PROM_NON_CACHED_REQS.inc()
            PROM_NON_CACHED_SPEED.observe(total_speed)
            PROM_NON_CACHED_REQS_CURRENT.dec()
        else:
            PROM_CACHED_REQS.inc()
            PROM_CACHED_SPEED.observe(total_speed)
            PROM_CACHED_REQS_CURRENT.dec()


class SurfNonCached(TaskSet):

    # report the number of users
    def on_start(self):
        PROM_NON_CACHED_USERS.inc()

    @PROM_NON_CACHED_REQS_CURRENT.track_inprogress()
    @task(1)
    def browse(self):
        logging.info('start non cached')

        PROM_INPUT_NON_CACHED_REQS.inc()
        headers = {'Cache-Control': 'no-cache'}

        config = CONFIG['non_cached']
        min = config['filesize']['min']
        max = config['filesize']['max']
        scaler = config['filesize']['scaler']
        filesize = genFileSize(min=min, max=max, scaler=scaler)
        fileid = genFileId(config['objects'])
        url = "/file/{}/{}".format(fileid, filesize)

        start = time.time()
        r = self.client.get(url, headers=headers)
        total_time = (time.time() - start)

        total_length = int(r.headers.get('content-length'))
        PROM_FILESIZE.observe(total_length)
        PROM_PROCESSED_NON_CACHED_REQS.inc()
        total_speed = total_length // total_time
        PROM_NON_CACHED_SPEED.observe(total_speed)


class SurfCached(TaskSet):

    # report the number of users
    def on_start(self):
        PROM_CACHED_USERS.inc()

    @PROM_CACHED_REQS_CURRENT.track_inprogress()
    @task(1)
    def browse(self):
        logging.info('start cached')
        PROM_INPUT_CACHED_REQS.inc()
        headers = {}

        config = CONFIG['cached']
        min = config['filesize']['min']
        max = config['filesize']['max']
        scaler = config['filesize']['scaler']
        filesize = genFileSize(min=min, max=max, scaler=scaler)
        fileid = genFileId(config['objects'])
        url = "/file/{}/{}".format(fileid, filesize)

        start = time.time()
        r = self.client.get(url, headers=headers)
        total_time = (time.time() - start)

        total_length = int(r.headers.get('content-length'))
        PROM_FILESIZE.observe(total_length)
        PROM_PROCESSED_CACHED_REQS.inc()
        total_speed = total_length // total_time
        PROM_CACHED_SPEED.observe(total_speed)


class FalseGet(TaskSet):
    # report the number of users
    def on_start(self):
        PROM_FALSE_USERS.inc()

    @task(1)
    def browse(self):
        logging.info('start false get')
        url = "/false"
        r = self.client.get(url, timeout=0.1)
        PROM_PROCESSED_FALSE_REQS.inc()

class UserNonCached(HttpLocust):
    config_noncached = CONFIG['non_cached']
    weight = config_noncached['weight']
    task_set = SurfNonCached
    min_wait = config_noncached['delay']['min']
    max_wait = config_noncached['delay']['max']


class UserCached(HttpLocust):
    config_cached = CONFIG['cached']
    weight = config_cached['weight']
    task_set = SurfCached
    min_wait = config_cached['delay']['min']
    max_wait = config_cached['delay']['max']


class UserFalse(HttpLocust):
    host = 'http://10.21.13.2:1234'
    weight = 1
    task_set = FalseGet
    min_wait = 0
    max_wait = 100


class RandomGet(HttpLocust):
    weight = 1
    task_set = RandomGetter
    min_wait = 0
    max_wait = 100