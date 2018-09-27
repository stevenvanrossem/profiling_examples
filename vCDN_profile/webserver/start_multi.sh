#! /bin/bash

#echo "start ssh"
#service ssh start

echo "start webserver server"

# delete default route (it is the docker0 interface)
ip route del default

# get the sap interface name (not 'eth0' and not 'lo')
intf=$(ifconfig -a | sed 's/[ \t].*//;/^\(lo\|\)$/d' | sed 's/[ \t].*//;/^\(eth0\|\)$/d')
echo "default interface: $intf"
ip route add default dev $intf

# start web server
bash rules.sh
python webserver8889.py > webserver1.log 2>&1 &
python webserver8890.py > webserver2.log 2>&1 &
python webserver8891.py > webserver4.log 2>&1 &
python webserver8892.py > webserver3.log 2>&1
