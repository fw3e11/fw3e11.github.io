---
title: 'Nginx, upstream and SSL'
date: 2017-03-01 15:12:18
tags:
  - Nginx
  - Web
  - SSL
---

### 一个最简单的反向代理
``` bash
$ sudo vi /etc/nginx/conf.d/upstream.conf

upstream timer {
    server  centos7-main:8000;
    server  centos7-slave1:8000;
    server  centos7-slave2:8000;
}
```
在`/etc/nginx/nginx.conf`里修改`server`即可：
``` conf
server {
    listen       80 default_server;
    listen       [::]:80 default_server;
    server_name  _;
    root         /usr/share/nginx/html;

    # Load configuration files for the default server block.
    include /etc/nginx/default.d/*.conf;

    location / {
        proxy_pass http://timer;
    }

    error_page 404 /404.html;
        location = /40x.html {
    }

    error_page 500 502 503 504 /50x.html;
        location = /50x.html {
    }
}
```

### 给HTTP加上S
制作私钥和CSR（Certificate Signing Request）证书请求文件
``` bash
$ sudo mkdir /etc/ssl/private
$ sudo chmod 700 /etc/ssl/private
$ openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout /etc/ssl/private/nginx-selfsigned.key -out /etc/ssl/certs/nginx-selfsigned.crt
$ openssl dhparam -out /etc/ssl/certs/dhparam.pem 2048
```
在nginx/conf.d/中加入ssl.conf
``` bash
$ sudo vi /etc/nginx/conf.d/ssl.conf

server {
    listen  443       http2 ssl;
    listen  [::]:443  http2 ssl;

    server_name _;

    location / {
        proxy_pass  http://timer;
    }

    ssl_certificate       /etc/ssl/certs/nginx-selfsigned.crt;
    ssl_certificate_key   /etc/ssl/private/nginx-selfsigned.key;
    ssl_dhparam           /etc/ssl/certs/dhparam.pem;
}
```
在nginx/default.d/中加入ssl-redirect.conf
``` bash
$ sudo vi /etc/nginx/default.d/ssl-redirect.conf

return 301 https://$host$request_uri;
```
