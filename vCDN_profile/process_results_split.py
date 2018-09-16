import yaml
import logging
import glob
import numpy as np
from execute_profiling import CONFIGS
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FactorAnalysis as FA
from sklearn import linear_model
from operator import add, div


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


def getProcessedResults(filepath):
    metrics = read_yaml(filepath)
    return metrics


def getMetricsFromFiles(filepath):

    metrics = {
        'wl_cachedPerc': [],
        'wl_delay': [],
        'wl_filesize': [],
        'wl_objects': [],
        'cpu@cache': [],
        'filesize@client': [],
        'cached_users@client': [],
        'non_cached_users@client': [],
        'false_users@client': [],
        'tx_byte_rate_cadv@cache:client': [],
        'rx_byte_rate_cadv@cache:server': [],
        'cached_download_latency@client': [],
        'non_cached_download_latency@client': [],
        'processed_cached_reqs_per_sec@client': [],
        'processed_non_cached_reqs_per_sec@client': [],
        'processed_false_reqs_per_sec@client': [],
    }

    resultList = glob.glob(filepath)
    # resultList = glob.glob("test_data1/results_*")
    resultsFolder = filepath.split('/')[0]

    for resultFile in resultList:
        filename = resultFile.split('/')[1]
        configId = filename.split('_')[2].split('.')[0]
        config = CONFIGS[configId]
        wl_cachedPerc = filename.split('_')[1]
        wl_delay = np.mean(config['delay'])
        min = config['filesize'][0]
        max = config['filesize'][1]
        scale = config['filesize'][2]
        wl_filesize = scale * (min + max) / 2
        objects = config['objects'][0]
        wl_objects = max * objects

        measurementList = read_yaml(resultFile)
        for run in measurementList:
            metrics['wl_cachedPerc'].append(wl_cachedPerc)
            metrics['wl_delay'].append(wl_delay)
            metrics['wl_filesize'].append(wl_filesize)
            metrics['wl_objects'].append(wl_objects)
            for metric in run['metrics']:
                name = metric.metric_name
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

    logging.info(metrics)

    resultspath = resultsFolder + '/' + 'processed.yml'
    write_yaml(resultspath, metrics)

    return metrics


if __name__ == "__main__":

    resultsfiles = "test_data1/results_*"
    resultsfiles2 = "50p_results/results_*"
    resultsfiles3 = "100Mb_results/results_*"
    resultsfiles4 = "1000Mb_results/results_*"
    resultsfiles5 = "0p_1000Mb_results/results_*"
    resultsfiles6 = "0p_results/results_*"

    # process metrics once
    #metrics = getMetricsFromFiles(resultsfiles)
    #metrics2 = getMetricsFromFiles(resultsfiles2)
    #metrics3 = getMetricsFromFiles(resultsfiles3)
    #metrics6 = getMetricsFromFiles(resultsfiles6)

    # get from already processed file
    metrics = getProcessedResults('test_data1/processed.yml')
    metrics2 = getProcessedResults('50p_results/processed.yml')
    #metrics = getProcessedResults('100Mb_results/processed.yml')
    #metrics2 = getProcessedResults('1000Mb_results/processed.yml')
    #metrics2 = getProcessedResults('0p_1000Mb_results/processed.yml')
    #metrics2 = getProcessedResults('0p_results/processed.yml')

    # calculate a custom metric
    metrics['total_users'] = list(map(add, metrics['cached_users@client'], metrics['non_cached_users@client']))
    metrics2['total_users'] = list(map(add, metrics2['cached_users@client'], metrics2['non_cached_users@client']))

    metrics['tx_byte_rate_perc@cache:client'] = np.divide(metrics['tx_byte_rate_cadv@cache:client'], 1)#2.5 #0.125
    metrics['rx_byte_rate_perc@cache:server'] = np.divide(metrics['rx_byte_rate_cadv@cache:server'], 1)#2.5
    metrics2['tx_byte_rate_perc@cache:client'] = np.divide(metrics2['tx_byte_rate_cadv@cache:client'], 1)
    metrics2['rx_byte_rate_perc@cache:server'] = np.divide(metrics2['rx_byte_rate_cadv@cache:server'], 1)

    metrics2['cpu@cache'] = np.multiply(metrics2['cpu@cache'], 1)

    metrics['cached_download_latency_perc@client'] = np.divide(metrics['cached_download_latency@client'], 1)#0.001
    metrics['non_cached_download_latency_perc@client'] = np.divide(metrics['non_cached_download_latency@client'], 1)#0.0025
    metrics2['cached_download_latency_perc@client'] = np.divide(metrics2['cached_download_latency@client'], 1)#0.05
    metrics2['non_cached_download_latency_perc@client'] = np.divide(metrics2['non_cached_download_latency@client'], 1)#0.1


    metricsInput = ['wl_cachedPerc', 'wl_delay', 'wl_filesize', 'wl_objects',
                    'cached_users@client', 'non_cached_users@client',
                    'false_users@client']

    metricsInput2 = [ #'wl_cachedPerc', 'total_users',
                      'cached_users@client',
                      'non_cached_users@client',
                      ]




    metricsOutput = ['cpu@cache', 'tx_byte_rate_cadv@cache:client', 'rx_byte_rate_cadv@cache:server',
                     'cached_download_latency@client', 'non_cached_download_latency@client',
                     'processed_cached_reqs_per_sec@client', 'processed_non_cached_reqs_per_sec@client',
                     'processed_false_reqs_per_sec@client']

    metricsOutput2 = [
                    'cpu@cache',
                    'cached_download_latency_perc@client', 'non_cached_download_latency_perc@client',
                    'tx_byte_rate_perc@cache:client', 'rx_byte_rate_perc@cache:server',
                    #'processed_cached_reqs_per_sec@client', 'processed_non_cached_reqs_per_sec@client',
                     ]

    # #singleScatter('non_cached_users@client', 'cpu@cache', metrics2)
    # singleScatter('cached_users@client', 'cached_download_latency@client', metrics)
    # singleScatter('cached_users@client', 'cached_download_latency@client', metrics2)

    inp = np.array([metrics[m] for m in metricsInput2]).T.astype(float)
    out = np.array([metrics[m] for m in metricsOutput2]).T.astype(float)

    all = np.concatenate((inp, out), axis=1)
    nSat = all[all[:, 2] <= 40]
    Sat = all[all[:, 2] > 40]
    inpSat = Sat[:, 0:2]
    outSat = Sat[:, 2:]
    inpnSat = nSat[:, 0:2]
    outnSat = nSat[:, 2:]

    scale = False
    ccanSat = CCA(n_components=1, scale=scale)
    ccanSat.fit(inpnSat, outnSat)
    inp_ccanSat = inpnSat.dot(ccanSat.x_weights_)
    out_ccanSat = outnSat.dot(ccanSat.y_weights_)
    plt.scatter(inp_ccanSat, out_ccanSat, c='orange', s=50)
    logging.info('ccanSat')
    logging.info(ccanSat.x_loadings_)
    logging.info(ccanSat.y_loadings_)
    print(ccanSat.score(inpnSat, outnSat))
    out_pred0 = inpnSat.dot(ccanSat.coef_[:, 0])
    plt.scatter(outnSat[:, 0], out_pred0, c='r', marker='+')


    ccaSat = CCA(n_components=1, scale=scale)
    ccaSat.fit(inpSat, outSat)
    inp_ccaSat = inpSat.dot(ccaSat.x_weights_)
    out_ccaSat = outSat.dot(ccaSat.y_weights_)
    plt.scatter(inp_ccaSat, out_ccaSat, c='purple')
    logging.info('ccaSat')
    #logging.info(ccaSat.x_rotations_)
    #logging.info(ccaSat.y_rotations_)


    # compare with second measurement
    inp2 = np.array([metrics2[m] for m in metricsInput2]).T.astype(float)
    out2 = np.array([metrics2[m] for m in metricsOutput2]).T.astype(float)

    n = 2 #5
    all = np.concatenate((inp2, out2), axis=1)
    nSat = all[all[:, n] <= 40]
    Sat = all[all[:, n] > 40]
    inpSat = Sat[:, 0:2]
    outSat = Sat[:, 2:]
    inpnSat = nSat[:, 0:2]
    outnSat = nSat[:, 2:]


    cca2Sat = CCA(n_components=1, scale=scale)
    cca2Sat.fit(inpSat, outSat)
    cca2nSat = CCA(n_components=1, scale=scale)
    cca2nSat.fit(inpnSat, outnSat)

    inp2_ccanSat = inpnSat.dot(ccanSat.x_weights_)
    out2_ccanSat = outnSat.dot(ccanSat.y_weights_)
    inp2_ccaSat = inpSat.dot(ccaSat.x_weights_)
    out2_ccaSat = outSat.dot(ccaSat.y_weights_)

    inp2_ccanSat = inpnSat.dot(cca2nSat.x_weights_)
    out2_ccanSat = outnSat.dot(cca2nSat.y_weights_)
    inp2_ccaSat = inpSat.dot(cca2Sat.x_weights_)
    out2_ccaSat = outSat.dot(cca2Sat.y_weights_)

    plt.scatter(inp2_ccanSat, out2_ccanSat, c='red')
    plt.scatter(inp2_ccaSat, out2_ccaSat, c='blue')

    plt.grid()
    plt.show()
    exit(0)

    #
    # # cached latency
    # hLat = all[all[:, 3] > 100]
    # lLat = all[all[:, 3] <= 0.15]
    # inpHLat = hLat[:, 0:2]
    # inpLLat = lLat[:, 0:2]
    # outHLat = hLat[:, 2:]
    # outLLat = lLat[:, 2:]
    #
    # # non cached latency
    # nhLat = all[all[:, 4] > 100]
    # nlLat = all[all[:, 4] <= 0.15]
    # inpnHLat = nhLat[:, 0:2]
    # inpnLLat = nlLat[:, 0:2]
    # outnHLat = nhLat[:, 2:]
    # outnLLat = nlLat[:, 2:]
    #
    # # linerate
    # hRate = all[all[:, 5] > 100]
    # inphRate = hRate[:, 0:2]
    # outhRate = hRate[:, 2:]

    #plt.scatter(inpnSat[:,1], outnSat[:,0], c='g')
    #plt.show()

    #inp = np.array([metrics[m] for m in metricsInput2]).T.astype(float)
    #out = np.array([metrics[m] for m in metricsOutput2]).T.astype(float)

    #singleScatter('cached_users@client', 'processed_cached_reqs_per_sec@client', metrics)
    #singleScatter('non_cached_users@client', 'processed_non_cached_reqs_per_sec@client', metrics)
    #singleScatter('wl_cachedPerc', 'cpu@cache', metrics)
    #singleScatter('cached_users@client', 'cpu@cache', metrics)
    #singleScatter('cached_download_latency@client', 'cpu@cache', metrics2)
    #singleScatter('non_cached_users@client', 'tx_byte_rate_cadv@cache:client', metrics2)
    #singleScatter('non_cached_users@client', 'tx_byte_rate_perc@cache:client', metrics)
    #singleScatter('non_cached_users@client', 'tx_byte_rate_perc@cache:client', metrics2)

    #exit(0)




    inp = np.array([metrics[m] for m in metricsInput2]).T.astype(float)
    out = np.array([metrics[m] for m in metricsOutput2]).T.astype(float)
    cca = CCA(n_components=1, scale=scale)
    cca.fit(inp, out)

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
    #print('Coefficients: \n', regr.coef_)

    inp_cca_n1 = inp.dot(cca.x_weights_)[:, 0]
    out_cca_n1 = out.dot(cca.y_weights_)[:, 0]
    #inp_cca_n2 = inp.dot(cca.x_weights_)[:, 1]
    #out_cca_n2 = out.dot(cca.y_weights_)[:, 1]

    plt.scatter(inp_cca_n1, out_cca_n1, c='g')
    #plt.scatter(inp_cca_n2, out_cca_n2, c='r')

    logging.info('cca1')
    logging.info(cca.x_rotations_)
    logging.info(cca.y_rotations_)


    #plt.grid()
    #plt.show()
    #exit(0)

    #plt.scatter(cca.x_scores_, cca.y_scores_, c='g')
    plt.plot(inp_cca, cca_regr, color='r', linewidth=0.5)



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
    cca2 = CCA(n_components=1, scale=scale)
    cca2.fit(inp2, out2)
    inp_cca2 = inp2.dot(cca.x_weights_)
    out_cca2 = out2.dot(cca.y_weights_)
    #inp_cca2 = inp2.dot(cca2.x_weights_)
    #out_cca2 = out2.dot(cca2.y_weights_)
    #inp_cca2 = cca2.x_scores_
    #out_cca2 = cca2.y_scores_
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(inp_cca2, out_cca2)
    cca_regr2 = regr.predict(inp_cca2)


    inp_cca2_n1 = inp2.dot(cca2.x_weights_)[:, 0]
    out_cca2_n1 = out2.dot(cca2.y_weights_)[:, 0]
    #inp_cca2_n2 = inp.dot(cca2.x_weights_)[:, 1]
    #out_cca2_n2 = out.dot(cca2.y_weights_)[:, 1]


    plt.scatter(inp_cca2_n1, out_cca2_n1, c='black', s=50)
    #plt.scatter(inp_cca2, out_cca2, c='beige', s=50)
    #plt.scatter(inp_cca2_n2, out_cca2_n2, c='grey', s=50)
    #plt.scatter(cca2.x_scores_, cca2.y_scores_, c='red', s=50)
    plt.plot(inp_cca2, cca_regr2, color='grey', linewidth=0.5)

    logging.info('cca2')
    logging.info(cca2.x_rotations_)
    logging.info(cca2.y_rotations_)

    #X_test = np.array([20, 20, 80])
    #cca_inp_test = X_test.dot(cca.x_rotations_)
    #cca_out_test = regr.predict(cca_inp_test.reshape(1, -1))[0]

    #plt.scatter(cca_inp_test, cca_out_test, s=100, c='b')



    # inp_cca_hLat = inpHLat.dot(cca2.x_weights_)
    # out_cca_hLat = outHLat.dot(cca2.y_weights_)
    # plt.scatter(inp_cca_hLat, out_cca_hLat, s=400, facecolors='none', edgecolors='red')
    #
    # inp_cca_nhLat = inpnHLat.dot(cca2.x_weights_)
    # out_cca_nhLat = outnHLat.dot(cca2.y_weights_)
    # plt.scatter(inp_cca_nhLat, out_cca_nhLat, s=500, facecolors='none', edgecolors='green')
    #
    # inp_cca_hRate = inphRate.dot(cca2.x_weights_)
    # out_cca_hRate = outhRate.dot(cca2.y_weights_)
    # plt.scatter(inp_cca_hRate, out_cca_hRate, s=100, facecolors='none', edgecolors='blue')
    #
    # inp_cca_sat = inpSat.dot(cca2.x_weights_)
    # out_cca_sat = outSat.dot(cca2.y_weights_)
    # plt.scatter(inp_cca_sat, out_cca_sat, s=150, facecolors='none', edgecolors='orange')



    plt.grid()
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






