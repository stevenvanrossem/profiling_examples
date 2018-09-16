import yaml
import subprocess
import shlex
import logging
import itertools
import requests
from time import sleep


logging.basicConfig(level=logging.DEBUG)

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


def set_config(filesize=[1,1,1], delay=[10,200], weights=[1,1], objects=[5,5], percentage_cached=50):
    configFile = 'client_config.yml'

    config = read_yaml(configFile)
    config['percentage_cached'] = percentage_cached
    config_nc = config['non_cached']
    config_c = config['cached']

    # filesize settings (size in MB, see webserver.py)
    min = filesize[0]
    max = filesize[1]
    scaler = filesize[2]
    config_nc['filesize']['min'] = min
    config_nc['filesize']['max'] = max
    config_nc['filesize']['scaler'] = scaler
    config_c['filesize']['min'] = min
    config_c['filesize']['max'] = max
    config_c['filesize']['scaler'] = scaler

    #delay settings
    min = delay[0]
    max = delay[1]
    config_nc['delay']['min'] = min
    config_nc['delay']['max'] = max
    config_c['delay']['min'] = min
    config_c['delay']['max'] = max
    config['delay']['min'] = min
    config['delay']['max'] = max

    #weight settings
    w_c = weights[0]
    w_nc = weights[1]
    config_c['weight'] = w_c
    config_nc['weight'] = w_nc

    #number of objects
    obj_c = objects[0]
    obj_nc = objects[1]
    config_c['objects'] = obj_c
    config_nc['objects'] = obj_nc

    write_yaml(configFile, config)
    # docker cp client_config.yml mn.client:/
    cmd = ['docker', 'cp', 'client_config.yml', 'mn.client:/']
    subprocess.check_call(cmd)


def reset_cache():
    # reset the cache docker host, so the memory starts from a clean slate
    # use the son-emu rest api for this
    url = 'http://localhost:5001/restapi/compute/dc1/cache'
    data = {
        "network": "(id=client,ip=10.10.0.1/24,mac=aa:aa:aa:00:00:01),"
                   "(id=server,ip=10.20.0.1/24,mac=aa:aa:aa:00:00:02)",
        "image": "squid-vnf"
           }
    resp = requests.delete(url)
    logging.info('delete cache: {0}'.format(resp))
    resp = requests.put(url, json=data)
    logging.info('restart cache: {0}'.format(resp))


def execute_run(conf, id):
    results_file = "results_" + str(conf) + "_" + str(id) + ".yml"
    # son-profile -p ped_squid.yml --mode passive --no-display -r result_num_users.yml
    cmd = ["son-profile", '-p', 'ped_squid.yml', '--mode', 'passive', '--no-display', '-r', results_file]
    cmd = "son-profile -p ped_squid.yml --mode passive --no-display -r " + results_file
    args = shlex.split(cmd)
    p = subprocess.Popen(args)
    p.wait()


def set_ratelimit(intf='dc1.s1-eth3', rate=1000):
    cmd = 'tcset --overwrite --device {0} --rate {1}M'.format(intf, rate)
    if rate == 0:
        cmd = 'tcdel --device {0} --all'.format(intf)
    args = shlex.split(cmd)
    p = subprocess.Popen(args)
    p.wait()



MEDIUM_CONF = {
    'id': 'medium',
    'delay': [1000, 2000],
    'filesize': [1, 5, 10], #min, max ,scaler(MB divider so 10 is 100kB)
    'objects': [400, 400] #cached, non_cached, 400 different files
}

CONFIGS = {'medium': MEDIUM_CONF}

if __name__ == "__main__":
    #cached_perc = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    #cached_perc = [10, 30, 70, 90]
    cached_perc = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    filesizes = [0.1, 0.5, 1, 5, 7, 10, 20, 30, 40, 50]
    #cached_perc = [10,90]
    #filesizes = [40]
    client_ratelimit = [1000]
    # calculate Cartesian product of all workload parameters
    configs = []
    for config in itertools.product(cached_perc, filesizes, client_ratelimit):
        configs.append(config)

    #confs = []
    #number_of_filesizes = 5
    #number_of_objects = 50
    #number_of_files = number_of_objects / number_of_filesizes

    #confs.append(MEDIUM_CONF)

    # test all possible workload configs
    for conf in configs:

        ratelimit = conf[2]
        filesize = conf[1]
        filesize2 = [filesize, filesize, 10]
        delay = [1000, 1000]
        objects = [50, 50]
        id = 'walltest6'

        cached_perc = conf[0]
        #c = float(cached_perc/10)
        #nc = float((100-cached_perc)/10)
        #weights = [c, nc]

        set_config(filesize=filesize2, delay=delay, percentage_cached=cached_perc, objects=objects)

        conf_str = '{0}_{1}_{2}'.format(cached_perc, filesize, ratelimit)
        reset_cache()
        #sleep(30)
        set_ratelimit(rate=ratelimit)
        execute_run(conf_str, id)
