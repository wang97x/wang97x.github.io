@echo off
setlocal

chcp 65001 >nul

set /p title=Please enter the post title:
if "%title%"=="" (
    echo Title cannot be empty.
    exit /b 1
)

set /p categories=Please enter categories (comma separated, default: study-notes):
if "%categories%"=="" set categories=study-notes

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
