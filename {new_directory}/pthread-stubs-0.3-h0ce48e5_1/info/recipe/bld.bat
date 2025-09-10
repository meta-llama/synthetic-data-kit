:: Trailing semicolon in this variable as set by current (2017/01)
:: conda-build breaks us. Manual fix:
set "MSYS2_ARG_CONV_EXCL=/AI;/AL;/OUT;/out"
:: Delegate to the Unixy script. We need to translate the key path variables
:: to be Unix-y rather than Windows-y, though.
set "saved_recipe_dir=%RECIPE_DIR%"
FOR /F "delims=" %%i IN ('cygpath.exe -u -p "%PATH%"') DO set "PATH_OVERRIDE=%%i"
FOR /F "delims=" %%i IN ('cygpath.exe -m "%LIBRARY_PREFIX%"') DO set "LIBRARY_PREFIX_M=%%i"
FOR /F "delims=" %%i IN ('cygpath.exe -u "%LIBRARY_PREFIX%"') DO set "LIBRARY_PREFIX_U=%%i"
FOR /F "delims=" %%i IN ('cygpath.exe -u "%PREFIX%"') DO set "PREFIX=%%i"
FOR /F "delims=" %%i IN ('cygpath.exe -u "%PYTHON%"') DO set "PYTHON=%%i"
FOR /F "delims=" %%i IN ('cygpath.exe -u "%RECIPE_DIR%"') DO set "RECIPE_DIR=%%i"
FOR /F "delims=" %%i IN ('cygpath.exe -u "%SP_DIR%"') DO set "SP_DIR=%%i"
FOR /F "delims=" %%i IN ('cygpath.exe -u "%SRC_DIR%"') DO set "SRC_DIR=%%i"
FOR /F "delims=" %%i IN ('cygpath.exe -u "%STDLIB_DIR%"') DO set "STDLIB_DIR=%%i"
bash -x %saved_recipe_dir%\build.sh
if errorlevel 1 exit 1
