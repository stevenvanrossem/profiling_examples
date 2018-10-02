import yaml
import logging
import glob
import numpy as np
from execute_profiling import CONFIGS
#import matplotlib.pyplot as plt
#from sklearn.cross_decomposition import CCA
#from sklearn.cross_decomposition import PLSRegression
#from sklearn.decomposition import FactorAnalysis as FA
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn import linear_model
#from operator import add, div
from copy import deepcopy
#import csv
import pandas as pd

logging.basicConfig(level='DEBUG')

def read_yaml(path):
    yml = None
    with open(path, "r") as f:
            try:
                yml = yaml.load(f)
            except yaml.YAMLError as ex:
                logging.exception("YAML error while reading %r." % path)
    return yml


def write_yaml(path, data):
    with open(path, "w") as f:
        try:
            yaml.dump(data, f, default_flow_style=False)
        except yaml.YAMLError as ex:
            logging.exception("YAML error while writing %r" % path)


def checkEqual(lst):
    return lst[1:] == lst[:-1]


def singleScatter(x_metric, y_metric, M):
    plt.scatter(M[x_metric], M[y_metric])
    plt.show()


def singleScatter2(x_index, y_index, M):
    plt.scatter(M[:, x_index], M[:, y_index])
    plt.show()


def getProcessedResults(filepath):
    metrics = read_yaml(filepath)
    return metrics


def getMetricsFromFiles(filepath):

    metrics = {
        'wl_cachedperc': [],
        #'wl_delay': [],
        'wl_filesize': [],
        #'wl_objects': [],
        'wl_ratelimit': [],
        'cpu_limit': [],
        'cpu@cache': [],
        'mem@cache': [],
        'disk_node@cache': [],
        'disk_write_speed@cache': [],
        'iowait_node_23@cache': [],
        'iowait_node_23_perc@cache': [],
        'idle_node_23@cache': [],
        'irq_node_23@cache': [],
        'softirq_node_23@cache': [],
        #'system_node_23@cache': [],
        #'user_node_23_perc@cache': [],
        'tcp_established_cadv@cache': [],
        'tasks_uninterruptible_cadv@cache': [],
        'filesize@client': [],
        'vcdn_users@client': [],
        #'cached_users@client': [],
        #'non_cached_users@client': [],
        #'false_users@client': [],
        'tx_bitrate_cadv@cache:client': [],
        'rx_bitrate_cadv@cache:server': [],
        'tx_packet_size_cadv@cache:client': [],
        'cached_download_latency@client': [],
        'non_cached_download_latency@client': [],
        'cached_download_time@client': [],
        'non_cached_download_time@client': [],
        #'processed_cached_reqs_per_sec@client': [],
        #'processed_non_cached_reqs_per_sec@client': [],
        #'processed_false_reqs_per_sec@client': [],
        'failed_reqs_per_sec@client': [],
    }

    ignored_metrics = [
        'cpu@server',
        'cpu@client',
        'disk@client',
        'disk@server',
    ]

    resultList = glob.glob(filepath)
    # resultList = glob.glob("test_data1/results_*")
    resultsFolder = filepath.split('/')[0]

    for resultFile in resultList:
        filename = resultFile.split('/')[1]
        #configId = filename.split('_')[2].split('.')[0]
        #config = CONFIGS[configId]
        wl_cachedPerc = filename.split('_')[1]
        wl_filesize = filename.split('_')[2]
        wl_ratelimit = filename.split('_')[3]
        cpu_limit = 100 #filename.split('_')[4]

        #wl_delay = np.mean(config['delay'])
        #min = config['filesize'][0]
        #max = config['filesize'][1]
        #scale = config['filesize'][2]
        #wl_filesize = scale * (min + max) / 2
        #objects = config['objects'][0]
        #wl_objects = max * objects

        measurementList = read_yaml(resultFile)
        for run in measurementList:
            metrics['wl_cachedperc'].append(wl_cachedPerc)
            #metrics['wl_delay'].append(wl_delay)
            metrics['wl_filesize'].append(wl_filesize)
            metrics['wl_ratelimit'].append(wl_ratelimit)
            metrics['cpu_limit'].append(cpu_limit)

            #metrics['wl_objects'].append(wl_objects)
            for metric in run['metrics']:
                #delta = abs(metric.CI.get('max',0) - metric.CI.get('min',0))
                #if metric.average * 2 < delta:
                #    value = metric.CI.get('max',0)
                #else:
                #    value = metric.average
                name = metric.metric_name
                if name in ignored_metrics:
                    continue
                #value = metric.median
                value = metric.average
                #print(name)
                #print(delta)
                #print(value)
                metrics[name].append(value)

        logging.info("got metrics from: " + resultFile)

    lenList = []
    for key, value in metrics.items():
        lenList.append(len(value))

    if checkEqual(lenList):
        logging.info("metrics are equal")
    else:
        logging.info("metrics are NOT equal")

    logging.info("metric names:")
    for key in metrics:
        logging.info(key + ', ')

    resultspath = resultsFolder + '/' + 'processed.yml'
    write_yaml(resultspath, metrics)

    df = pd.DataFrame.from_dict(metrics)

    resultspath = resultsFolder + '/' + 'processed.csv'
    df.to_csv(resultspath)


    #return metrics





resultsfiles = "test_data1/results_*"
resultsfiles2 = "50p_results/results_*"
resultsfiles3 = "100Mb_results/results_*"
resultsfiles4 = "1000Mb_results/results_*"
resultsfiles5 = "0p_1000Mb_results/results_*"
resultsfiles6 = "0p_results/results_*"

resultsfiles7 = "1024MB_1core/results_*"
resultsfiles8 = "1024MB_1core_extended/results_*"
resultsfiles9 = "walltest1/results_*"
resultsfiles9 = "test_wall/results_*"
resultsfiles9 = "walltest2/results_*"
resultsfiles9 = "walltest3/results_*"
resultsfiles9 = "walltest4/results_*"
resultsfiles9 = "walltest6/results_*"
resultsfiles9 = "walltest_nodiskio/results_*"
resultsfiles9 = "walltest_nodiskio_2/results_*"

# process metrics once
#metrics = getMetricsFromFiles(resultsfiles)
#metrics2 = getMetricsFromFiles(resultsfiles2)
#metrics3 = getMetricsFromFiles(resultsfiles3)
#metrics6 = getMetricsFromFiles(resultsfiles6)

getMetricsFromFiles(resultsfiles9)
