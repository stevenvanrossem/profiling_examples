FROM python:slim
MAINTAINER steven.vanrossem@intec.ugent.be

RUN	apt-get update && apt-get install -y \
	net-tools \
	iproute \
	inetutils-ping \
        ethtool \
	curl \
        nano \
        psmisc \
        procps


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
ADD locustfile2.py locustfile2.py
RUN chmod +x locustfile2.py
ADD locustfile3.py locustfile3.py
RUN chmod +x locustfile3.py
ADD locustfile4.py locustfile4.py
RUN chmod +x locustfile4.py
ADD locustfile5.py locustfile5.py
RUN chmod +x locustfile5.py
ADD locustfile6.py locustfile6.py
RUN chmod +x locustfile6.py
ADD locustfile7.py locustfile7.py
RUN chmod +x locustfile7.py
ADD locustfile8.py locustfile8.py
RUN chmod +x locustfile8.py
ADD locustfile9.py locustfile9.py
RUN chmod +x locustfile9.py
ADD locustfile10.py locustfile10.py
RUN chmod +x locustfile10.py
ADD fill_cache.py fill_cache.py
RUN chmod +x fill_cache.py

ADD client_config.yml client_config.yml
RUN chmod +x client_config.yml

ADD start_client.sh start_client.sh
RUN chmod +x start_client.sh
ADD start_client2.sh start_client2.sh
RUN chmod +x start_client2.sh


CMD /bin/bash
