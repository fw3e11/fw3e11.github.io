---
title: TCP/IP Illustrated Vol.1 (1)
date: 2017-04-18 22:08:43
tags:
  - TCP/IP
  - Network
---

最近开始做二级代理的相关工作，发现自己对`TCP/IP`这个协议知之甚少，很多地方模棱两可。正视自己的不足，开始学习《TCP/IP协议详解 卷一：协议》。

### 4层协议
#### 分层
|||||
|:-----:|:-----:|:--:|:-----:|
| 应用层 | Application | handles the details of particular application | Telnet, FTP, SMTP, etc
| 运输层 | Transport | provides a flow of data between two computers | TCP, UDP
| 网络层 | Network | handles the movement of packets | IP, ICMP, IGMP
| 链路层 | Link | handles hardware details of physically interfacing | device driver and interface card

#### TCP/IP层
网络层`IP`提供了一个不可靠的服务，在不保证可靠性的情况下尽可能快地把包`packet`从源点移动到终点。
> In the TCP/IP protocol suite the network layer, IP, provides an unreliable service. That is, it does its best job of moving a packet from its source to its final destination, but there are no guarantees.

`TCP`在不可靠的IP层上提供一个可靠的服务，为了提供这种可靠的服务，TCP采用了超时重传、发送/接受端到端的确认等（机制）。
> TCP provides a reliable transport layer, even though the service it uses (IP) is unreliable.
> To provide this service, TCP performs timeout and retransmission, sends and receives end-to-end acknowledgments, and so on.

`UDP`发送和接受`数据报`（一个信息单元从发送方传输到接收方）。UDP是不可靠的，不能保证数据报最终到达它的目的地。
> UDP sends and receives datagrams for applications. A datagram is a unit of information that travels from the sender to the receiver. Unlike TCP, however, UDP is unreliable. There is no guarantee that the datagram ever gets to its final destination.

`IP`是网络层上的主要协议。TCP和UDP都使用IP，很少有应用程序直接使用IP。`ICMP`是`IP`的附属协议，我们常用的`ping`和`traceroute`就是使用了ICMP。`IGMP`(Internet Group Management Protocol)用于多播：把UDP数据报发送给多个主机。

`ARP`(Address Resolution Protocol)和`RARP`(Reverse ARP)用于网络层和链路层之间，针对某些特定的网络接口，用来转换IP地址和网络接口地址。

#### IP地址
> Every interface on an internet must have a unique Internet address (also called an IP address)

IP地址长度为32位（32bits），也就是4字节（4bytes）。每个字节用一个十进制整数来表示，这就是“Dotted decimal notation”。IP地址分为5类，ABCDE，区分方法体现在高位上第一个字节位：以二进制表示，0开头的是A类，10开头的是B类，110是C类，1110是D类，1111是E类。

#### Encapsulation 封装
数据在发送过程中自顶向下经过每层协议栈，都会附加一些信息，名字也会有变化。
TCP发送给IP的数据单元叫`TCP段`，IP发送给网络接口的数据单元叫`IP数据报` ，通过以太网传输的比特流叫做`帧`。
> When an application sends data using TCP, the data is sent down the protocol stack, through each layer, until it is sent as a stream of bits across the network. Each Layer adds information to the data by prepending headers (and sometimes add tailer information) to the data that it receives.
  * The unit of data that TCP sends to IP is called a `TCP segment`.
  * The unit of data that IP sends to the network interface is called an `IP datagram`. (To be completely accurate, called a `packet` that can either be an `IP datagram` or `a fragment of an IP datagram`)
  * The stream of bits that flows acorss the Ethernet is called a `frame`.
  * The only differences between TCP and UDP are that the unit of data that UDP passes to IP is called a `UDP datagram`, and the size of UDP header is less (UDP header is 8 bytes whereas TCP header is 20 bytes).

#### 端口号
> TCP and UDP identify applications using 16-bit port numbers.

Math.pow(2, 16) - 1 等于65535。
