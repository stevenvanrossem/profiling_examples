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

csvFile = '1024MB_1core/processed.csv'
metrics = pd.read_csv(csvFile, index_col=0)
print(metrics.columns.values)

metricsInput = [ 'wl_cachedperc',
                  'vcdn_users',
                  'wl_filesize',
                  ]

metricsOutput2 = [
                'cached_download_latency@client',
                'non_cached_download_latency@client',
                #'cached_download_bw@client',
                #'processed_cached_reqs_per_sec@client', 'processed_non_cached_reqs_per_sec@client',
                #'cpu@cache',
                #'tx_byte_rate_cadv@cache:client',
                #'rx_byte_rate_perc@cache:server',
                 ]