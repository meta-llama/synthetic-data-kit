



pip -h
IF %ERRORLEVEL% NEQ 0 exit /B 1
pip list
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0
