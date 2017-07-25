---
title: LittleProxy Study Notes
date: 2017-07-25 17:33:17
tags:
  - Java
  - Netty
  - Network
---
> LittleProxy is a high performance HTTP proxy written in Java atop Trustin Lee's excellent [Netty](http://netty.io) event-based networking library. It's quite stable, performs well, and is easy to integrate into your projects.

## Hello-World-Like Tutorial
``` Java
HttpProxyServer server = DefaultHttpProxyServer.bootstrap()
    .withPort(8080)
    .start();
```

## More Complicated
源码结构解析：

``` Java
/*
 * DefaultHttpProxyServer.java
 * class DefaultHttpProxyServer implements HttpProxyServer
 * Primary implementation of an {@link HttpProxyServer}.
 */
public static HttpProxyServerBootstrap bootstrap();

/*
 * HttpProxyServerBootstrap.java
 * interface HttpProxyServerBootstrap
 * Configures and starts an {@link HttpProxyServer}.
 */
HttpProxyServerBootstrap withFiltersSource(HttpFiltersSource filtersSource);

/*
 * HttpFiltersSource.java
 * interface HttpFiltersSource
 * Factory for {@link HttpFilters}.
 */
 HttpFilters filterRequest(HttpRequest originalRequest, ChannelHandlerContext ctx);
 /*
  * If the request/response size exceeds the maximum buffer size,
  * the request/response will fail, and
  * the proxy will stop processing the request/respond with a 502 Bad Gateway error.
  */
 int getMaximumRequestBufferSizeInBytes();
 int getMaximumResponseBufferSizeInBytes();

 /*
  * HttpFilters.java
  * interface HttpFilters
  *
  * Multiple methods are defined, corresponding to different steps in the request processing lifecycle.
  * Because HTTP transfers can be chunked, for any given request or response, the filter methods (that can modify request/response in place) may be called multiple times:
  * 1. once for the initial {@link HttpRequest} or {@link HttpResponse}
  * 2. once for each subsequent {@link HttpContent}
  * 3. the last chunk will always be a {@link LastHttpContent} and can be checked for being last using {@link ProxyUtils#isLastChunk(HttpObject)}.
  */
  public interface HttpFilters {}
```

#### HttpFilters
由此引入我们第一个介绍的接口`HttpFilters`，该接口定义了如下方法，在使用时会按照先后顺序被调用：

1. clientToProxyRequest
2. proxyToServerConnectionQueued
3. proxyToServerResolutionStarted
4. proxyToServerResolutionSucceeded
5. proxyToServerRequest (can be multiple if chunked)
6. proxyToServerConnectionStarted
7. proxyToServerConnectionFailed (if connection couldn't be established)
8. proxyToServerConnectionSSLHandshakeStarted (only if HTTPS required)
9. proxyToServerConnectionSucceeded
10. proxyToServerRequestSending
11. proxyToServerRequestSent
12. serverToProxyResponseReceiving
13. serverToProxyResponse (can be multiple if chunked)
14. serverToProxyResponseReceived
15. proxyToClientResponse

以下是一个通过代理请求打印信息的例子：
``` Java
public class CustomFiltersSource extends HttpFiltersSourceAdapter {

  @Override
  public HttpFilters filterRequest(HttpRequest originalRequest, ChannelHandlerContext ctx) {
    return new UserDefinedHttpFilters(originalRequest, ctx);
  }

  private class UserDefinedHttpFilters extends HttpFiltersAdapter {

    UserDefinedHttpFilters(HttpRequest originalRequest, ChannelHandlerContext ctx) {
      super(originalRequest, ctx);
    }

    @Override
    public HttpResponse clientToProxyRequest(HttpObject httpObject) {
      printInfo("clientToProxyRequest", httpObject);
      return super.clientToProxyRequest(httpObject);
    }

    @Override
    public HttpResponse proxyToServerRequest(HttpObject httpObject) {
      printInfo("proxyToServerRequest", httpObject);
      return super.proxyToServerRequest(httpObject);
    }

    private void printInfo(String methodName, HttpObject httpObject) {
      System.out.println(methodName + ": ");
      System.out.println("httpObject is last: " + ProxyUtils.isLastChunk(httpObject));
      originalRequest.headers().forEach(entry -> System.out.println(entry.getKey() + ": " + entry.getValue()));
      System.out.println(originalRequest.getProtocolVersion().toString() + " " + originalRequest.getUri());
      System.out.println();
    }
  }
}
```

``` Java
HttpProxyServer server = DefaultHttpProxyServer.bootstrap()
    .withFiltersSource(new CustomFiltersSource())
    .withPort(8080)
    .start();
```

#### ChainedProxy
接口`ChainedProxy`在且仅在`ChainedProxyManager`中使用，`ChainedProxyManager`也是一个接口，提供`lookupChainedProxies`方法。一般情况下可以通过继承`ChainedProxyAdapter`的方式使用，不需要直接实现`ChainedProxy`。
``` Java
/**
 * Interface for classes that manage chained proxies.
 */
public interface ChainedProxyManager {
  /**
   * 1. Based on the given httpRequest, add any {@link ChainedProxy}s to the list that should be used to process the request. The downstream proxy will attempt to connect to each of these in the order that they appear until it successfully connects to one.
   * 2. To allow the proxy to fall back to a direct connection, you can add {@link ChainedProxyAdapter#FALLBACK_TO_DIRECT_CONNECTION} to the end of the list.
   * 3. To keep the proxy from attempting any connection, leave the list blank. This will cause the proxy to return a 502 response.
   */
  void lookupChainedProxies(HttpRequest httpRequest, Queue<ChainedProxy> chainedProxies);
}
```

#### ActivityTracker
> Interface for receiving information about activity in the proxy.

#### FlowContext & FullFlowContext
> Encapsulates contextual information for flow information that's being reported to a {@link ActivityTracker}.

> Extension of {@link FlowContext} that provides additional information (which we know after actually processing the request from the client).

#### ProxyAuthenticator
Interface for objects that can authenticate someone for using our Proxy on the basis of a username and password.
