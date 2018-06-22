import yaml
import subprocess
import shlex
import logging



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
            LOG.exception("YAML error while writing %r" % path)


def set_config(filesize=[1,1,1], delay=[10,200], weights=[1,1], objects=[5,5]):
    configFile = 'client_config.yml'

    config = read_yaml(configFile)
    config_nc = config['non_cached']
    config_c = config['cached']

    # filesize settings
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

def execute_run(cached_perc, id):
    results_file = "100p_results_" + str(cached_perc) + "_" + str(id) + ".yml"
    # son-profile -p ped_squid.yml --mode passive --no-display -r result_num_users.yml
    cmd = ["son-profile", '-p', 'ped_squid.yml', '--mode', 'passive', '--no-display', '-r', results_file]
    cmd = "son-profile -p ped_squid.yml --mode passive --no-display -r " + results_file
    args = shlex.split(cmd)
    p = subprocess.Popen(args)
    p.wait()

MEDIUM_CONF = {
    'id': 'medium',
    'delay': [1000, 2000],
    'filesize': [1, 5, 1], #min, max ,scaler
    'objects': [10, 10] #cached, non_cached
}

CONFIGS = {'medium': MEDIUM_CONF}

if __name__ == "__main__":
    cached_perc = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    cached_perc = [10, 30, 70, 90]

    confs = []
    number_of_filesizes = 5
    number_of_objects = 50
    number_of_files = number_of_objects / number_of_filesizes

    confs.append(MEDIUM_CONF)

    for conf in confs:
        filesize = conf['filesize']
        delay = conf['delay']
        objects = conf['objects']
        id = conf['id']
        for p in cached_perc:
            c = float(p/10)
            nc = float((100-p)/10)
            weights = [c, nc]

            set_config(filesize=filesize, delay=delay, weights=weights, objects=objects)

            execute_run(p, id)
