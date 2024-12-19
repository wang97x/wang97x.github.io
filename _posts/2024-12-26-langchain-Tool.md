---
layout: post 
title: Langchain相关概念-工具
author: "wang"
date: 2024-12-19 18:26:00 +0800
categories: [Langchain]
tags: [Conceptual]
---
# 工具
## 概述
LangChain 中的 tool 抽象将 Python 函数 与定义函数 名称、描述 和 预期参数 的 模式 相关联。

工具 可以传递给支持 工具调用 的 聊天模型，允许模型请求使用特定输入执行特定函数。

## 关键概念
- 工具是一种封装函数及其模式的方式，该模式可以传递给聊天模型。
- 使用 @tool 装饰器创建工具，这简化了工具创建过程，并支持以下功能：
    - 自动推断工具的 名称、描述 和 预期参数，同时还支持自定义。
    - 定义返回 工件（例如图像、数据帧等）的工具
    - 使用 注入的工具参数 从模式中（因此从模型中）隐藏输入参数。

## 工具接口
工具接口在 BaseTool 类中定义，它是 Runnable 接口 的子类。

与工具模式相对应的关键属性
```
name：工具的名称。
description：工具的功能描述。
args：返回工具参数的 JSON 模式的属性。
```
执行与工具关联的函数的关键方法
```
invoke：使用给定的参数调用工具。
ainvoke：使用给定的参数异步调用工具。用于 Langchain 的异步编程。
```

## 使用 @tool 装饰器创建工具
创建工具的推荐方法是使用 @tool 装饰器。此装饰器旨在简化工具创建过程，在大多数情况下应使用它。定义函数后，可以使用 @tool 对其进行装饰，以创建实现 工具接口 的工具。
```
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
   """Multiply two numbers."""
   return a * b
```

## 直接使用工具
定义工具后，可以通过调用函数直接使用它。例如，要使用上面定义的 multiply 工具
```
multiply.invoke({"a": 2, "b": 3})
```

## 