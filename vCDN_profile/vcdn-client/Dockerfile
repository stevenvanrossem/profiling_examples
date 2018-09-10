FROM python:slim
MAINTAINER steven.vanrossem@intec.ugent.be

RUN	apt-get update && apt-get install -y \
	net-tools \
	iproute \
	inetutils-ping \
        ethtool \
	curl \
        nano


ADD start.sh start.sh
RUN chmod +x start.sh

WORKDIR /

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


ADD http_proxy_setting /http_proxy_setting
RUN chmod +rwx http_proxy_setting

# set entry point for emulator gatekeeper
ENV SON_EMU_CMD "./start.sh"

ADD squid_client.py squid_client.py
RUN chmod +x squid_client.py

ADD locustfile.py locustfile.py
RUN chmod +x locustfile.py

ADD client_config.yml client_config.yml
RUN chmod +x client_config.yml

ADD start_client.sh start_client.sh
RUN chmod +x start_client.sh


CMD /bin/bash
