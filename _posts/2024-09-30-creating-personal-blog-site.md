---
layout: post
title: "使用Jekyll搭建个人博客"
date: 2024-09-30 22:56:51 +0800
categories: [Blogging]
tags: [jekyll, github pages]
---
本文将介绍如何使用Jekyll搭建个人博客，并部署在GitHub Pages上。


## 1.搭建个人博客
下面正式开始搭建个人博客网站。参考：[Chirpy - Getting Started](https://chirpy.cotes.page/posts/getting-started/)。

### 1.1 创建网站
打开[chirpy-starter](https://github.com/cotes2020/chirpy-starter)仓库，点击按钮 "Use this template" → "Create a new repository"。

![create-repository-step1](/assets/images/creating-personal-blog-site/create-repository-step1.png)

将新仓库命名为`<username>.github.io`，其中`<username>`是你的GitHub用户名，如果包含大写字母需要转换为小写。

![create-repository-step2](/assets/images/creating-personal-blog-site/create-repository-step2.png)

注：如果不需要自定义主题样式，则推荐使用这种方式，因为容易升级，并且能隔离无关文件，使你能够专注于文章内容写作。

### 1.2 部署
[GitHub Pages](https://pages.github.com/)是一个通过GitHub托管和发布网页的服务，官方文档：<https://docs.github.com/en/pages>。本文使用GitHub Pages部署个人博客网站。

每个GitHub用户可以创建一个用户级网站，仓库名为`<username>.github.io`，发布地址为 `https://<username>.github.io`。GitHub Pages支持自定义域名，参考文档[About custom domains and GitHub Pages](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/about-custom-domains-and-github-pages)。

在部署之前，检查_config.yml中的`url`是否正确配置为上述发布地址（或者自定义域名）。

> 注意：一般不需要配置`baseurl`。如果配置了，则文章中必须使用`relative_url`过滤器生成正确的URL，否则会导致404错误。参考[Jekyll’s site.url and baseurl](https://mademistakes.com/mastering-jekyll/site-url-baseurl/)。
{: .prompt-warning }

之后在GitHub上打开仓库设置，点击左侧导航栏 "Pages"，在 "Build and deployment" - "Source" 下拉列表选择 "GitHub Actions"。

![github-pages-deployment-source](/assets/images/creating-personal-blog-site/github-pages-deployment-source.png)

提交本地修改并推送至远程仓库，将会触发Actions工作流。在仓库的Actions标签页将会看到 "Build and Deploy" 工作流正在运行。构建成功后，即可通过配置的URL访问自己的博客网站。

<https://vaxh.github.io/>

![personal-blog-site](/assets/images/creating-personal-blog-site/personal-blog-site.png)
