---
title: TCP/IP Illustrated Vol.1 (3) -- Ping and Traceroute
date: 2017-04-29 08:34:37
tags:
  - TCP/IP
  - Network
---

### Ping
> The ping program sends an ICMP echo request to a host, expecting an ICMP echo reply to be returned.
Normally if you can't Ping a host, you won't be able to Telnet or FTP to that host. Conversely, if you can't Telnet to a host, Ping is often the starting point to determine what the problem is.
Ping also measures the round-trip time to the host, give us some indication of how "far away" that host is.
