---
title: 'TCP/IP Illustrated Vol.1 (2) -- IP: Internet Protocol'
date: 2017-04-19 23:38:52
tags:
  - TCP/IP
  - Network
---
> IP is the workhorse protocol of the TCP/IP protocol suite. All TCP, UDP, ICMP, and IGMP data gets transmitted as IP datagrams. A fact that amazes many newcomers to TCP/IP is that IP provides an unreliable, connectionless datagram delivery service.
 * By **unreliable** we mean there are no guarantees that an IP datagram successfully gets to its destination. IP provides a best effort service. When something goes wrong, ..., IP has a simple error handling algorithm: throw away the datagram and try to send an ICMP message back to the source. Any required reliability must be provided by the upper layers (e.g., TCP).
 * The term **connectionless** means that IP does not maintain any state information about successive datagrams. Each datagram is handled independently from all other datagrams. This also means that IP datagrams can get delivered out of order.

### IP header
![IP Header](/img/IP-Header.png)
* `4-bit version` (0-15).
  * 当前版本是4，因此IP有时又称为IPv4。
* `4-bit header length` (0-15), the number of 32-bit words in the header.
  * 每32位字长记为1，`IP header`最小长度为20字节，因此`header length`字段最小为5；`header length`最大为15，因此`IP header`最长为60字节。
* `8-bit type of service (TOS)`, is composed 3-bit precedence field (which is ignored today), 4 TOS bits, and an unused bit that must be 0.
  * 所以只需要关心其中4位即可，和tcpdump的输出有关。
* `16-bit total length (in bytes)` (0-65535).
* `16-bit identification` (0-65535), uniquely identifies each datagram sent by a host. It normally increments by 1 each time a datagram is sent.
* `8-bit time-to-live (TTL)` (0-255), an upper limit on the number of routers through which a datagram can pass. It is initialized by the sender (often 32 or 64) and decremented by 1 by every router that handles the datagram. When this field reaches 0, the datagram is thrown away, and the sender is notified with an ICMP message.
  * traceroute
* `8-bit protocol` (0-255), identifies which protocol gave the data for IP to send.
* `16-bit header checksum`, is calculated over the IP header only. It does not cover any data follows the header.

  >To compute the checksum for an outgoing datagram, the value of the checksum field is first set to 0. Then the 16-bit one's complement sum of the header is calculated (i.e., the entire header is considered a sequence of 16-bit words). The 16-bit one's complement of this sum is stored in the checksum field.

  `IP header`看做16位字长的序列，checksum算法就是将所有16位字的序列以补码形式相加，然后再对相加和取补。

  > When an IP datagram is received, the 16-bit one's complement sum of the header is calculated. Since the receiver's calculated checksum contains the checksum stored by the sender, the receiver's checksum is all 1 bits if nothing in header was modified. If not (a checksum error), IP discards the received datagram. No error message is generated. It is up to higher layers to somehow detect the missing datagram and retransmit.

  由于接收方计算的checksum包含了sender发送的checksum，<del>根据某种数学原理</del>，接收方计算的checksum应该全为1，否则就是出错了。如果出错，IP抛弃收到的数据报，而非生成一个错误消息，由上层发现丢失数据报并重传。
* `32-bit source address` and `32-bit destination address`

### IP routing
IP层既可以配置成路由器，也可以配置成主机。他两个的本质区别是路由器转发数据报，而主机不转发（除非被特殊设置）。
> IP layer can be configured to act as a router in addition to acting as a host. The fundamental difference is that a host __never__ forwards datagrams from one of its interfaces to another, while a router forwards datagrams. A host that contains embedded router functionality should never forward a datagram unless it has been specifically configured to do so.

IP层接收并发送的数据有两种：一种是本地生成的数据报，另一种是从网络接口中接收待转发的数据报。
> IP can receive a datagram from TCP, UDP, ICMP, or IGMP (that is, a locally generated datagram) to send, or one that has been received from a network interface (a datagram to forward).

IP层在内存中有一个路由表，每次接收到数据报都要搜索这个路由表。当IP层接收来自网络接口的数据报，首先要检查（数据报的）目的IP地址是否本机的IP地址之一，或者是否是IP广播地址：如果是、这个数据报会被送到IP header中protocol字段所指定的协议模块进行处理，如果不是、(1)如果IP层被设置为路由，则进行转发；(2)否则（如果是主机）则数据报被抛弃。
> The IP layer has a routing table in memory that it searches each time it receives a datagram to send. When a datagram is received from a network interface, IP first checks if the destination IP address is one of its own IP addresses or an IP broadcase address. If so, the datagram is delivered to the protocol module specified by the protocol field in the IP header. If the datagram is not destined for this IP layer, the (1) if the IP layer was configured to act as a router the packet is forwarded, else (2) the datagram is silently discarded.

路由表中的每一项都包含：
1. 目的IP地址
2. 下一站路由的IP地址
3. 标志（网络地址or主机地址，下一站路由是真的下一站路由or直连接口）
4. 为数据报的传输指定一个网络接口

IP路由选择主要是搜索路由表：
1. 寻找与目的IP地址匹配的表目
2. 寻找与目的网络号相匹配的表目
3. 寻找默认表目

如果以上3个都没有成功，则该数据报不能被传送，同时如果该数据报来自本地，那么一般会向生成数据报的应用程序返回一个“主机不可达”或“网络不可达”的错误。

IP路由选择机制的另一个基本特性是：为一个网络指定一个路由，而不必为每个主机指定一个路由。这样可以极大地缩小路由表的规模
> The ability to specify a route to a network, and not have to specify a route to every host, is another fundamental feature of IP routing. Doing this allows the routers on the Internet, for example, to have a routing table with thousands of entries, instead of a routing table with millions of entries.

### Subnet Addressing
上面说的**为一个网络指定一个路由**，我的理解这一个个的网络就是子网。如果把IP简单划分为网络号和主机号，A类网络（2<sup>24</sup>-2）和B类网络（2<sup>16</sup>-2）就拥有了太多的主机。这样IP层的路由表就会非常的大。

拿B类网络来说，一般默认情况是把16位主机号一分为二，前8位用于子网号，后8位用作主机号。这样一来，所有以140.252开头的B类IP地址，只需要一个路由表目。

子网对外部路由隐藏了内部的网络结构。换种说法，这一台外部路由负责了内部所有子网和主机的Internet接入。比如一个`IP目的地址`为`140.252.57.1`的数据报，先到达`140.252.104.1`的`gateway`（外部网关），这个`路由网关`需要知道`子网号`是57的路由表目，中间可能不能直达57号子网，需要通过55号子网作为中转而后“跳”到57号子网。

### Subnet Mask
子网掩码是一个32位长度的地址，它的作用结合IP地址，将IP地址划分为网络地址和主机地址。

子网掩码标记为1的留给网络号和子网号，标记为0的留给主机号。

### ipconfg
``` bash
ifconfig -a
```

### netstat
``` bash
netstat -in
```
