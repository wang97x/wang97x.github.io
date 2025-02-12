@echo off
setlocal

:: Set code page to UTF-8
chcp 65001 >nul

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
set current_date=%year%-%month%-%day% %hour%:%minute%:%second% +0800

:: Ask for the file name suffix
set /p userinput=Please enter the file name suffix:

:: Construct the full file name as year-month-day-userinput.md
set filename=%year%-%month%-%day%-%userinput%.md

:: Create a new file based on the template
(
    echo ---
    echo layout: post
    echo title: "%userinput%"
    echo author: "wang"
    echo date: %current_date%
    echo categories: [blog]
    echo tags: []
    echo ---
) > "%filename%"

:: Inform user that the file was created
echo File "%filename%" has been created with the current date: %current_date%

:: Open the file with Typora
start typora "%filename%"

endlocal
