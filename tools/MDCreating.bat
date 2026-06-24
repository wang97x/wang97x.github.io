@echo off
setlocal

chcp 65001 >nul

set /p title=Please enter the post title:
if "%title%"=="" (
    echo Title cannot be empty.
    exit /b 1
)

set /p categories=Please enter topic categories (comma separated, e.g. AI Agent/大模型与训练/检索与排序/工具与框架/编程基础/博客建设):
if "%categories%"=="" (
    echo Categories cannot be empty.
    exit /b 1
)

set /p tags=Please enter tags (comma separated, optional):

set script_path=%~dp0new-post.ps1

if not exist "%script_path%" (
    echo Cannot find script: "%script_path%"
    exit /b 1
)

set "command=powershell -ExecutionPolicy Bypass -File ""%script_path%"" -Title ""%title%"" -CategoriesCsv ""%categories%"""

if not "%tags%"=="" (
    set "command=%command% -TagsCsv ""%tags%"""
)

for /f "usebackq delims=" %%F in (`%command%`) do set created_file=%%F

if not defined created_file (
    echo Failed to create post.
    exit /b 1
)

echo File "%created_file%" has been created.
start typora "%created_file%"

endlocal
