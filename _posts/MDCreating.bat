@echo off
setlocal

:: Get current date and time using wmic
for /f "tokens=1,2 delims=." %%a in ('wmic os get localdatetime ^| find "."') do (
    set datetime=%%a
)

:: Extract date and time components
set year=%datetime:~0,4%
set month=%datetime:~4,2%
set day=%datetime:~6,2%
set hour=%datetime:~8,2%
set minute=%datetime:~10,2%
set second=00 

:: Combine date and time into the Jekyll format
set datetime=%year%-%month%-%day% %hour%:%minute%:%second% +0800

:: Ask for the filename
set /p filename=Please enter the file name: %year%-%month%-%day%-

:: Create the Markdown file with Jekyll-compliant YAML front matter
echo --- > "%filename%.md"
echo layout: post >> "%filename%.md"
echo title: "New Document" >> "%filename%.md"
echo author: "wang" >> "%filename%.md"
echo date: %datetime% >> "%filename%.md"
echo categories: [blog] >> "%filename%.md"
echo tags: [blog] >> "%filename%.md"
echo --- >> "%filename%.md"
echo # First line >> "%filename%.md"

:: Inform user that the file was created
echo File "%filename%.md" has been created

:: Open the file with Typora
start typora "%filename%.md"

endlocal
