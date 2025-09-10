:: Create and enter the build directory
md build && cd build

:: Build shared library
cmake -GNinja ^
      -DCMAKE_INSTALL_PREFIX="%LIBRARY_PREFIX%" ^
      -DCMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS=FALSE ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DBUILD_SHARED_LIBS=ON ^
      ..
if errorlevel 1 exit /b 1

ninja && ninja install
if errorlevel 1 exit /b 1
