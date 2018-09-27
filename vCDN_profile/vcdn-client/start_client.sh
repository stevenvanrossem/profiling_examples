#!/bin/bash

# kill previous locust
pkill -9 -f locust

# set env
export http_proxy="10.10.0.1:3128"

# start 2 slaves (problem with prometheus, only one slave seems to be exporting)
#locust -f locustfile.py --slave --master-host=localhost&
#locust -f locustfile2.py --slave --master-host=localhost&
#locust -f locustfile3.py --slave --master-host=localhost&

# start master
#locust --master --expect-slaves=2 --host http://10.20.0.2:8888 --no-web -c $1 -r 100
locust -f locustfile.py --host http://10.20.0.2:8888 --no-web -c $1 -r 100 CDNUser&
locust -f locustfile2.py --host http://10.20.0.2:8888 --no-web -c $1 -r 100 CDNUser&
locust -f locustfile3.py --host http://10.20.0.2:8888 --no-web -c $1 -r 100 CDNUser&
locust -f locustfile4.py --host http://10.20.0.2:8888 --no-web -c $1 -r 100 CDNUser&
locust -f locustfile5.py --host http://10.20.0.2:8888 --no-web -c $1 -r 100 CDNUser
