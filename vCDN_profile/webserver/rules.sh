#!/bin/bash

iptables -F
iptables -F -t nat

# Always accept packets for already-active sessions.
iptables -A INPUT -m state --state RELATED,ESTABLISHED -j ACCEPT

# Default input policy is to ACCEPT
iptables -P INPUT ACCEPT

# One in the remaining two go to port 3128
iptables -t nat -A PREROUTING -p tcp --dport 8888 -m statistic --mode nth --every 4 --packet 0 -j REDIRECT --to-port 8889
iptables -t nat -A PREROUTING -p tcp --dport 8888 -m statistic --mode nth --every 3 --packet 0 -j REDIRECT --to-port 8890
iptables -t nat -A PREROUTING -p tcp --dport 8888 -m statistic --mode nth --every 2 --packet 0 -j REDIRECT --to-port 8891
# The remaining one always routes to port 3129
iptables -t nat -A PREROUTING -p tcp --dport 8888 -j REDIRECT --to-port 8892


