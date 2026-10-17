@echo off
rem Compatibility launcher; canonical name uses snake_case.
call "%~dp0run_video.cmd" %*
exit /b %errorlevel%
