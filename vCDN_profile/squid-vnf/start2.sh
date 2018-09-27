#! /bin/bash

echo "start squid server"
sysctl -w net.ipv4.conf.all.proxy_arp=1
bash rules.sh 
/usr/sbin/squid3 -d 5 -f squid1.conf &
/usr/sbin/squid3 -d 5 -f squid2.conf



