import yaml
import logging
import glob
import numpy as np
from execute_profiling import CONFIGS
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import FactorAnalysis as FA
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from operator import add, div
from copy import deepcopy
import csv
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
        'wl_cachedPerc': [],
        #'wl_delay': [],
        'wl_filesize': [],
        #'wl_objects': [],
        'wl_ratelimit': [],
        'cpu_limit': [],
        'cpu@cache': [],
        'mem@cache': [],
        'filesize@client': [],
        'vcdn_users@client': [],
        #'cached_users@client': [],
        #'non_cached_users@client': [],
        #'false_users@client': [],
        'tx_bitrate_cadv@cache:client': [],
        'rx_bitrate_cadv@cache:server': [],
        'cached_download_latency@client': [],
        'non_cached_download_latency@client': [],
        #'processed_cached_reqs_per_sec@client': [],
        #'processed_non_cached_reqs_per_sec@client': [],
        #'processed_false_reqs_per_sec@client': [],
    }

    ignored_metrics = [
        'cpu@server',
        'cpu@client'
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
            metrics['wl_cachedPerc'].append(wl_cachedPerc)
            #metrics['wl_delay'].append(wl_delay)
            metrics['wl_filesize'].append(wl_filesize)
            metrics['wl_ratelimit'].append(wl_ratelimit)
            metrics['cpu_limit'].append(cpu_limit)

            #metrics['wl_objects'].append(wl_objects)
            for metric in run['metrics']:
                name = metric.metric_name
                if name in ignored_metrics:
                    continue
                value = metric.median
                metrics[name].append(value)

        logging.info("got metrics from: " + resultFile)

    lenList = []
    for key, value in metrics.iteritems():
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

# process metrics once
#metrics = getMetricsFromFiles(resultsfiles)
#metrics2 = getMetricsFromFiles(resultsfiles2)
#metrics3 = getMetricsFromFiles(resultsfiles3)
#metrics6 = getMetricsFromFiles(resultsfiles6)

getMetricsFromFiles(resultsfiles7)

# get from already processed file
#metrics = getProcessedResults('test_data1/processed.yml')
metrics2 = getProcessedResults('50p_results/processed.yml')
metrics3 = getProcessedResults('100Mb_results/processed.yml')
#metrics2 = getProcessedResults('1000Mb_results/processed.yml')
#metrics2 = getProcessedResults('0p_1000Mb_results/processed.yml')
metrics = getProcessedResults('0p_results/processed.yml')

# calculate a custom metric
metrics['total_users'] = list(map(add, metrics['cached_users@client'], metrics['non_cached_users@client']))
metrics2['total_users'] = list(map(add, metrics2['cached_users@client'], metrics2['non_cached_users@client']))

metrics['tx_byte_rate_perc@cache:client'] = np.divide(metrics['tx_byte_rate_cadv@cache:client'], 2.5)#2.5 #0.125
metrics['rx_byte_rate_perc@cache:server'] = np.divide(metrics['rx_byte_rate_cadv@cache:server'], 2.5)#2.5
metrics2['tx_byte_rate_perc@cache:client'] = np.divide(metrics2['tx_byte_rate_cadv@cache:client'], 2.5)
metrics2['rx_byte_rate_perc@cache:server'] = np.divide(metrics2['rx_byte_rate_cadv@cache:server'], 2.5)
metrics3['tx_byte_rate_perc@cache:client'] = np.divide(metrics3['tx_byte_rate_cadv@cache:client'], 2.5)
metrics3['rx_byte_rate_perc@cache:server'] = np.divide(metrics3['rx_byte_rate_cadv@cache:server'], 2.5)



metrics['cached_download_latency_perc@client'] = np.divide(metrics['cached_download_latency@client'], 0.001)#0.001
metrics['non_cached_download_latency_perc@client'] = np.divide(metrics['non_cached_download_latency@client'], 0.001)#0.0025
metrics['cached_download_bw@client'] = np.divide(1, metrics['cached_download_latency@client'])#0.001
metrics['non_cached_download_bw@client'] = np.divide(1, metrics['non_cached_download_latency@client'])#0.0025

metrics2['cached_download_latency_perc@client'] = np.divide(metrics2['cached_download_latency@client'], 0.001)#0.05
metrics2['non_cached_download_latency_perc@client'] = np.divide(metrics2['non_cached_download_latency@client'], 0.001)#0.1
metrics3['cached_download_latency_perc@client'] = np.divide(metrics3['cached_download_latency@client'], 0.001)#0.05
metrics3['non_cached_download_latency_perc@client'] = np.divide(metrics3['non_cached_download_latency@client'], 0.001)#0.1



metricsInput = ['wl_cachedPerc', 'wl_delay', 'wl_filesize', 'wl_objects',
                'cached_users@client', 'non_cached_users@client',
                'false_users@client']

metricsInput2 = [ 'wl_cachedPerc',
                  'total_users',
                  #'cached_users@client',
                  #'non_cached_users@client',
                  ]




metricsOutput = ['cpu@cache', 'tx_byte_rate_cadv@cache:client', 'rx_byte_rate_cadv@cache:server',
                 'cached_download_latency@client', 'non_cached_download_latency@client',
                 'processed_cached_reqs_per_sec@client', 'processed_non_cached_reqs_per_sec@client',
                 'processed_false_reqs_per_sec@client']

metricsOutput2 = [
                'cached_download_latency@client',
                'non_cached_download_latency@client',
                #'cached_download_bw@client',
                #'processed_cached_reqs_per_sec@client', 'processed_non_cached_reqs_per_sec@client',
                #'cpu@cache',
                #'tx_byte_rate_cadv@cache:client',
                #'rx_byte_rate_perc@cache:server',
                 ]

def doCCA(metrics, color):

    inp = np.array([metrics[m] for m in metricsInput2]).T.astype(float)
    out = np.array([metrics[m] for m in metricsOutput2]).T.astype(float)
    inp0 = np.zeros(len(metricsInput2))
    out0 = np.zeros(len(metricsOutput2))
    inp = np.vstack((inp, inp0))
    out = np.vstack((out, out0))

    all = np.concatenate((inp, out), axis=1)
    # fixed cache
    fixed = all[all[:, 0] == 90]
    inp_fixed = fixed[:, 1:2]
    out_fixed = fixed[:, 2:6]
    #singleScatter2(1, 2, fixed)
    #singleScatter2(1, 3, fixed)
    # singleScatter2(1, 4, fixed)
    # singleScatter2(1, 5, fixed)

    inp = inp_fixed #inpnSat #inp_fixed
    out = out_fixed #outnSat #out_fixed

    poly = PolynomialFeatures(1, include_bias=False, interaction_only=False)
    inp = poly.fit_transform(inp)
    # inp = inp_poly[:, 2:]

    cca = CCA(n_components=1, scale=False)
    cca.fit(inp, out)
    print(cca.score(inp, out))
    inp_cca = inp.dot(cca.x_rotations_)
    out_cca = out.dot(cca.y_rotations_)

    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(inp_cca, out_cca)
    cca_regr = regr.predict(inp_cca)
    # The coefficients
    print('Coefficients: \n', regr.coef_)

    plt.scatter(inp_cca, out_cca, c=color)
    plt.plot(inp_cca, cca_regr, color=color, linewidth=0.5)

    logging.info('cca')
    logging.info(cca.x_loadings_)
    logging.info(cca.y_loadings_)
    logging.info(cca.coef_)
    return cca.coef_

def doPLS(metrics, color='r', marker='+', perc=10):
    inp0 = np.zeros(len(metricsInput2))
    out0 = np.zeros(len(metricsOutput2))

    inp = np.array([metrics[m] for m in metricsInput2]).T.astype(float)
    out = np.array([metrics[m] for m in metricsOutput2]).T.astype(float)
    inp = np.vstack((inp, inp0))
    out = np.vstack((out, out0))

    all = np.concatenate((inp, out), axis=1)
    # fixed cache
    fixed = all[all[:, 0] == perc]
    inp_fixed = fixed[:, 1:2]
    out_fixed = fixed[:, 2:4]
    #singleScatter2(1, 2, fixed)
    #singleScatter2(1, 3, fixed)
    # singleScatter2(1, 4, fixed)
    # singleScatter2(1, 5, fixed)

    inp = inp_fixed #inpnSat #inp_fixed
    out = out_fixed #outnSat #out_fixed

    poly = PolynomialFeatures(1, include_bias=False, interaction_only=False)
    inp = poly.fit_transform(inp)
    # inp = inp_poly[:, 2:]

    pls2 = PLSRegression(n_components=1, scale=False)
    pls2.fit(inp, out)
    print(pls2.score(inp, out))
    print(pls2.coef_)
    out_pls_pred0 = inp.dot(pls2.coef_)[:, 0]
    #plt.scatter(inp[:, 0], inp.dot(pls2.coef_)[:, 0], c='r', marker=marker)
    #plt.scatter(inp[:, 0], pls2.predict(inp)[:, 0], c='r', marker=marker)
    plt.scatter(inp[:, 0], out[:, 0], c='black', s=30, marker=marker)
    #plt.scatter(inp[:, 0], out[:, 1], c='grey', s=30, marker=marker)
    #plt.scatter(inp[:, 0], inp.dot(pls2.coef_)[:, 1], c='g', marker=marker)
    #plt.scatter(inp[:, 0], pls2.predict(inp)[:, 1], c='g', marker=marker)
    #plt.scatter(inp[:, 0], out[:, 1], c='black', marker=marker)

    return pls2.coef_, out


if __name__ == "__main__":


    #metrics3 = deepcopy(metrics)
    #metrics3['cpu@cache'] = np.multiply(metrics3['cpu@cache'], 2)

    #coef1 = doCCA(metrics, 'green')
    #coef2 = doCCA(metrics2, 'red')
    #ratio = np.divide(coef1, coef2)
    #print(ratio)
    # #doCCA(metrics3, 'blue')
    #plt.grid()
    # #plt.savefig('test1.png')
    #plt.show()
    # exit(0)

    coef1, out1 = doPLS(metrics, marker='o', perc=90)
    coef2, out2 = doPLS(metrics2, marker='+', perc=90)
    ratio = np.divide(coef1, coef2)
    print(ratio)
    ratio = np.divide(out1, out2)
    print(ratio)
    #doCCA(metrics3, 'blue')
    plt.grid()
    #plt.savefig('test1.png')
    plt.show()
    exit(0)


    # inp = np.array([metrics[m] for m in metricsInput2]).T.astype(float)
    # out = np.array([metrics[m] for m in metricsOutput2]).T.astype(float)
    # all = np.concatenate((inp, out), axis=1)
    #
    # FA = FA()
    # FA.fit(all)
    # print(FA.get_covariance())


    #plt.scatter(inpnSat[:,1], outnSat[:,0], c='g')
    #plt.show()

    #inp = np.array([metrics[m] for m in metricsInput2]).T.astype(float)
    #out = np.array([metrics[m] for m in metricsOutput2]).T.astype(float)


    #singleScatter('non_cached_users@client', 'non_cached_download_latency@client', metrics)
    #singleScatter('cached_users@client', 'cached_download_latency@client', metrics2)
    #singleScatter('cached_users@client', 'processed_cached_reqs_per_sec@client', metrics)
    #singleScatter('non_cached_users@client', 'processed_non_cached_reqs_per_sec@client', metrics)
    #singleScatter('wl_cachedPerc', 'cpu@cache', metrics)
    #singleScatter('cached_users@client', 'cpu@cache', metrics)
    #singleScatter('cached_users@client', 'non_cached_download_bw@client', metrics)
    #singleScatter('cpu@cache', 'non_cached_download_latency@client', metrics)
    #singleScatter('cpu@cache', 'tx_byte_rate_cadv@cache:client', metrics)
    #singleScatter('non_cached_users@client', 'tx_byte_rate_perc@cache:client', metrics)
    #singleScatter('non_cached_users@client', 'tx_byte_rate_perc@cache:client', metrics2)

    #exit(0)


    inp0 = np.zeros(len(metricsInput2))
    out0 = np.zeros(len(metricsOutput2))

    scale = False
    inp = np.array([metrics[m] for m in metricsInput2]).T.astype(float)
    out = np.array([metrics[m] for m in metricsOutput2]).T.astype(float)
    inp = np.vstack((inp, inp0))
    out = np.vstack((out, out0))

    all = np.concatenate((inp, out), axis=1)
    # fixed cache
    fixed = all[all[:, 0] == 10]
    inp_fixed = fixed[:, 1:2]
    out_fixed = fixed[:, 2:4]
    singleScatter2(1, 2, fixed)
    singleScatter2(1, 3, fixed)
    #singleScatter2(1, 4, fixed)
    #singleScatter2(1, 5, fixed)


    # cached latency
    all = fixed
    Sat = all[all[:, 4] > 80]
    nSat = all[all[:, 4] <= 80]
    inpSat = Sat[:, 1:2]
    inpnSat = nSat[:, 1:2]
    outSat = Sat[:, 2:4]
    outnSat = nSat[:, 2:4]

    inp = inp_fixed #inpnSat #inp_fixed
    out = out_fixed #outnSat #out_fixed

    poly = PolynomialFeatures(2, include_bias=False, interaction_only=False)
    inp = poly.fit_transform(inp)
    # inp = inp_poly[:, 2:]

    cca = CCA(n_components=1, scale=True)
    cca.fit(inp, out)
    #out_pred = cca.predict(inp)
    #print(cca.score(inp, out))
    inp_cca = inp.dot(cca.x_rotations_)
    out_cca = out.dot(cca.y_rotations_)
    print(cca.x_loadings_)
    print(cca.y_loadings_)
    plt.scatter(inp_cca, out_cca, c='purple')


    pls2 = PLSRegression(n_components=1, scale=True)
    pls2.fit(inp, out)
    print(pls2.score(inp, out))
    print(pls2.x_loadings_)
    print(pls2.y_loadings_)
    inp_pls = inp.dot(pls2.x_rotations_)
    out_pls = out.dot(pls2.y_rotations_)
    #plt.scatter(pls2.x_scores_, pls2.y_scores_, c='g')
    #plt.scatter(inp_pls, out_pls, c='g')

    pls2 = PLSRegression(n_components=1, scale=scale)
    pls2.fit(inp, out)
    print(pls2.score(inp, out))
    out_pls_pred0 = inp.dot(pls2.coef_)[:, 0]
    plt.scatter(out, inp.dot(pls2.coef_), c=['r', 'g'], marker='+')
    #out_pls_pred1 = inp.dot(pls2.coef_[:, 1])
    #plt.scatter(out[:, 1], out_pls_pred1, c='grey', marker='+')

    #plt.scatter(inp[:, 0], out[:, 0] - out_pls_pred0, c='grey', marker='+')
    #plt.scatter(inp[:, 1], out[:, 1] - out_pls_pred1, c='green', marker='+')

    inp_pls = inp.dot(pls2.x_rotations_)
    out_pls = out.dot(pls2.y_rotations_)
    #plt.scatter(inp_pls, out_pls, c=['r'])
    #plt.scatter(pls2.x_scores_, pls2.y_scores_, c='g')



    plt.grid()
    plt.show()
    exit(0)

    inp_cca = inp.dot(cca.x_weights_)
    out_cca = out.dot(cca.y_weights_)
    #inp_cca = cca.x_scores_
    #out_cca = cca.y_scores_
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(inp_cca, out_cca)
    cca_regr = regr.predict(inp_cca)
    # The coefficients
    print('Coefficients: \n', cca.coef_)

    inp_cca_n1 = inp.dot(cca.x_weights_[:, 0])
    out_cca_n1 = out.dot(cca.y_weights_[:, 0])
    out_cca_pred = inp.dot(cca.coef_[:, 0])
    #out_cca_pred2 = inp.dot(cca.coef_[:, 1])
    #out_cca_pred3 = inp.dot(cca.coef_[:, 2])
    #out_cca_pred4 = inp.dot(cca.coef_[:, 3])
    # inp_cca_n2 = inp.dot(cca.x_weights_[:, 1])
    # out_cca_n2 = out.dot(cca.y_weights_[:, 1])

    #plt.scatter(out_cca_pred, out[:, 0], c='g', marker='+')

    #plt.scatter(inp_cca_n1, out_cca_n1, c='g')
    #plt.scatter(inp_cca_n1, out_cca_pred, c='grey', marker='+')
    #plt.scatter(inp_cca_n1, out_cca_pred2, c='r', marker='+')
    #plt.scatter(inp_cca_n1, out_cca_pred3, c='b', marker='+')
    #plt.scatter(inp_cca_n1, out_cca_pred4, c='purple', marker='+')
    #plt.scatter(cca.x_scores_, cca.y_scores_, c='r')
    plt.scatter(inp_cca_n1, out_cca_n1, c='grey')
    #plt.scatter(inp, out_pred, c='r')

    logging.info('cca1')
    logging.info(cca.x_rotations_)
    logging.info(cca.y_rotations_)


    plt.grid()
    plt.show()
    exit(0)

    #plt.scatter(cca.x_scores_, cca.y_scores_, c='g')
    plt.plot(inp_cca, cca_regr, color='g', linewidth=0.5)



    # X_meas = inp[0, :]
    # Y_meas = out[0, :]
    # cca_in = X_meas.dot(cca.x_rotations_)
    # cca_in2 = inp_cca[0]
    # cca_out = Y_meas.dot(cca.y_rotations_)
    # cca_out2 = out_cca[0]
    # cca_out3 = regr.predict(cca_in.reshape(1, -1))[0]

    #plt.scatter(cca_in, cca_out, s=100, c='g')
    #plt.scatter(cca_in, cca_out3, s=100, c='g', marker='+')

    # X_measb = inp[10, :]
    # Y_measb = out[10, :]
    # cca_inb = X_measb.dot(cca.x_rotations_)
    # cca_outb = regr.predict(cca_inb.reshape(1, -1))[0]
    #
    # xratio = cca_inb / cca_in
    # yratio = Y_measb / Y_meas


    #plt.scatter(cca_inb, cca_outb, s=500, c='g', marker='+')


    # also plot second cca test
    inp2 = np.array([metrics2[m] for m in metricsInput2]).T.astype(float)
    out2 = np.array([metrics2[m] for m in metricsOutput2]).T.astype(float)
    inp2 = np.vstack((inp2, inp0))
    out2 = np.vstack((out2, out0))
    cca2 = CCA(n_components=1, scale=scale)
    cca2.fit(inp2, out2)
    inp_cca2 = inp2.dot(cca.x_weights_)
    out_cca2 = out2.dot(cca.y_weights_)
    #inp_cca2 = inp2.dot(cca2.x_weights_)
    #out_cca2 = out2.dot(cca2.y_weights_)
    #inp_cca2 = cca2.x_scores_
    #out_cca2 = cca2.y_scores_



    inp_cca2_n1 = inp2.dot(cca2.x_weights_)
    out_cca2_n1 = out2.dot(cca2.y_weights_)
    #inp_cca2_n2 = inp.dot(cca2.x_weights_)[:, 1]
    #out_cca2_n2 = out.dot(cca2.y_weights_)[:, 1]
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(inp_cca2_n1, out_cca2_n1)
    cca_regr2 = regr.predict(inp_cca2_n1)
    # The coefficients
    print('Coefficients: \n', regr.coef_)

    #plt.scatter(inp_cca2_n1, out_cca2_n1, c='black', s=50)
    plt.scatter(inp_cca2, out_cca2, c='beige', s=50)
    #plt.scatter(inp_cca2_n2, out_cca2_n2, c='grey', s=50)
    #plt.scatter(cca2.x_scores_, cca2.y_scores_, c='red', s=50)
    #plt.plot(inp_cca2_n1, cca_regr2, color='grey', linewidth=0.5)

    logging.info('cca2')
    logging.info(cca2.x_rotations_)
    logging.info(cca2.y_rotations_)

    # plt.grid()
    # plt.savefig('test2.png')
    # plt.show()
    # exit(0)

    # inp3 = np.array([metrics3[m] for m in metricsInput2]).T.astype(float)
    # out3 = np.array([metrics3[m] for m in metricsOutput2]).T.astype(float)
    # cca3 = CCA(n_components=1, scale=scale)
    # cca3.fit(inp3, out3)
    # inp_cca3 = inp3.dot(cca3.x_weights_)
    # out_cca3 = out3.dot(cca3.y_weights_)
    # plt.scatter(inp_cca3, out_cca3, c='purple', s=50)
    #
    # inp_cca3b = inp2.dot(cca3.x_weights_)
    # out_cca3b = out2.dot(cca3.y_weights_)
    # plt.scatter(inp_cca3b, out_cca3b, c='orange', s=100, alpha=0.5)
    #
    # logging.info('cca3')
    # logging.info(cca3.x_rotations_)
    # logging.info(cca3.y_rotations_)

    #X_test = np.array([20, 20, 80])
    #cca_inp_test = X_test.dot(cca.x_rotations_)
    #cca_out_test = regr.predict(cca_inp_test.reshape(1, -1))[0]

    #plt.scatter(cca_inp_test, cca_out_test, s=100, c='b')


    all = np.concatenate((inp, out), axis=1)
    # cached latency
    hLat = all[all[:, 2] > 90]
    lLat = all[all[:, 2] <= 90]
    inpHLat = hLat[:, 0:2]
    inpLLat = lLat[:, 0:2]
    outHLat = hLat[:, 2:]
    outLLat = lLat[:, 2:]

    inp_cca_hLat = inpHLat.dot(cca.x_weights_)
    out_cca_hLat = outHLat.dot(cca.y_weights_)
    plt.scatter(inp_cca_hLat, out_cca_hLat, s=150, facecolors='none', edgecolors='red')

    # # non cached latency
    nhLat = all[all[:, 3] > 90]
    nlLat = all[all[:, 3] <= 90]
    inpnHLat = nhLat[:, 0:2]
    inpnLLat = nlLat[:, 0:2]
    outnHLat = nhLat[:, 2:]
    outnLLat = nlLat[:, 2:]

    inp_cca_nhLat = inpnHLat.dot(cca.x_weights_)
    out_cca_nhLat = outnHLat.dot(cca.y_weights_)
    plt.scatter(inp_cca_nhLat, out_cca_nhLat, s=50, facecolors='none', edgecolors='green')




    all = np.concatenate((inp2, out2), axis=1)
    # cached latency
    hLat = all[all[:, 2] > 90]
    lLat = all[all[:, 2] <= 90]
    inpHLat = hLat[:, 0:2]
    inpLLat = lLat[:, 0:2]
    outHLat = hLat[:, 2:]
    outLLat = lLat[:, 2:]

    inp_cca_hLat = inpHLat.dot(cca2.x_weights_)
    out_cca_hLat = outHLat.dot(cca2.y_weights_)
    plt.scatter(inp_cca_hLat, out_cca_hLat, s=150, facecolors='none', edgecolors='red')

    # # non cached latency
    nhLat = all[all[:, 3] > 90]
    nlLat = all[all[:, 3] <= 90]
    inpnHLat = nhLat[:, 0:2]
    inpnLLat = nlLat[:, 0:2]
    outnHLat = nhLat[:, 2:]
    outnLLat = nlLat[:, 2:]

    inp_cca_nhLat = inpnHLat.dot(cca2.x_weights_)
    out_cca_nhLat = outnHLat.dot(cca2.y_weights_)
    plt.scatter(inp_cca_nhLat, out_cca_nhLat, s=100, facecolors='none', edgecolors='green')



    #
    # inp_cca_hRate = inphRate.dot(cca2.x_weights_)
    # out_cca_hRate = outhRate.dot(cca2.y_weights_)
    # plt.scatter(inp_cca_hRate, out_cca_hRate, s=100, facecolors='none', edgecolors='blue')
    #
    # inp_cca_sat = inpSat.dot(cca2.x_weights_)
    # out_cca_sat = outSat.dot(cca2.y_weights_)
    # plt.scatter(inp_cca_sat, out_cca_sat, s=150, facecolors='none', edgecolors='orange')



    plt.grid()

    plt.savefig('test.png')
    plt.show()


    #x_scatter = cca.x_scores_
    #y_scatter = cca.y_scores_

    x_scatter2 = inp.dot(cca.x_weights_)
    y_scatter2 = out.dot(cca.y_weights_)

    #plt.ion()


    #plt.scatter(x_scatter, y_scatter, c='r')

    #plt.scatter(x_scatter2, y_scatter2, c='g')
    #plt.show()

    # x_scatter = cca.x_scores_[:, 1]
    # y_scatter = cca.y_scores_[:, 1]
    # plt.scatter(x_scatter, y_scatter)
    #plt.show()






