FROM python:slim
MAINTAINER steven.vanrossem@intec.ugent.be

RUN	apt-get update && apt-get install -y \
	net-tools \
	iproute \
	inetutils-ping \
        ethtool \
        procps \
        nano \
        iptables


ADD start_single.sh start_single.sh
RUN chmod +x start_single.sh
ADD start_multi.sh start.sh
RUN chmod +x start.sh
ADD rules.sh rules.sh
RUN chmod +x rules.sh

WORKDIR /
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


ADD web_server.py web_server.py
RUN chmod +x web_server.py
ADD webserver8889.py webserver8889.py
RUN chmod +x webserver8889.py
ADD webserver8890.py webserver8890.py
RUN chmod +x webserver8890.py
ADD webserver8891.py webserver8891.py
RUN chmod +x webserver8891.py
ADD webserver8892.py webserver8892.py
RUN chmod +x webserver8892.py

# set entry point for emulator gatekeeper
ENV SON_EMU_CMD "./start.sh"

CMD /bin/bash
