import yaml

def readConfig(path):
    yml = None
    with open(path, "r") as f:
            try:
                yml = yaml.load(f)
            except yaml.YAMLError as ex:
                logging.exception("YAML error while reading %r." % path)
    return yml


CONFIG = readConfig("client_config.yml")

cached = CONFIG.get('cached')