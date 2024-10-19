---
layout: post 
title: "ElasticSearch"
author: "wang"
date: 2024-10-11 14:57:00 +0800
categories: [LLM]
tags: [llm, ElasticSearch]
---
# ElasticSearch
## ES概述
ES是一个开源的高扩展的分布式全文检索引擎，它可以近乎实时的存储、检索数据；本身扩展性很好，可以扩展到上百台服务器，处理PB级别（大数据时代）的数据。ES也使用Java开发并使用Lucene作为其核心来实现所有索引和搜索的功能，但是它的目的是通过简单的RestFul API来隐藏Lucene的复杂性，从而让全文检索变得简单。现今，ES已经是全世界排名第一的搜索引擎类应用！

## 安装
1. 下载地址
https://www.elastic.co/cn/downloads/elasticsearch
2. 安装包解压
3. 运行elasticSearch.bat
4. 修改配置
```yaml
# Enable security features
xpack.security.enabled: false

xpack.security.enrollment.enabled: true

# Enable encryption for HTTP API client connections, such as Kibana, Logstash, and Agents
xpack.security.http.ssl:
  enabled: false
```



## 索引库操作



## 文档操作