FROM ubuntu:14.04
MAINTAINER steven.vanrossem@intec.ugent.be

RUN apt-get update && apt-get clean
RUN apt-get install -y squid3 && apt-get clean
RUN apt-get install -y nano && apt-get clean
RUN apt-get install -y iptables && apt-get clean
ADD squid.conf /etc/squid3/squid.conf
RUN mkdir /var/cache/squid
ADD squid.conf squid.conf
ADD squid1.conf squid1.conf
ADD squid2.conf squid2.conf
RUN chown proxy:proxy /var/cache/squid
RUN /usr/sbin/squid3 -N -z -F

ADD rules.sh rules.sh
RUN chmod +x rules.sh
ADD start.sh start.sh
RUN chmod +x start.sh
ADD start2.sh start2.sh
RUN chmod +x start2.sh

#EXPOSE 3128
#EXPOSE 3129
#EXPOSE 3127

# set entry point for emulator gatekeeper
#ENV SON_EMU_CMD "/usr/sbin/squid3 -d 5"
#ENV SON_EMU_CMD "./start2.sh"
ENV SON_EMU_CMD "./start.sh"

#CMD /usr/sbin/squid3 -N -d 0
CMD /bin/bash

