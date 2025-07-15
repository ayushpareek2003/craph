@echo off
echo CRAPH GPU Graph Traversal Library Build Script
echo ==============================================

REM Check if CUDA is available
nvcc --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: CUDA not found. Please install CUDA Toolkit.
    echo Make sure nvcc is in your PATH.
    pause
    exit /b 1
)

echo CUDA found. Building library...

REM Create directories
if not exist "obj" mkdir obj
if not exist "obj\kernels" mkdir obj\kernels
if not exist "lib" mkdir lib

REM Build CPU components
echo Building CPU components...
g++ -std=c++17 -Wall -Wextra -O2 -fPIC -I./inc -c src/sparse_structs.cpp -o obj/sparse_structs.o
if errorlevel 1 (
    echo ERROR: Failed to compile CPU components.
    pause
    exit /b 1
)

REM Build CUDA components
echo Building CUDA components...
nvcc -std=c++17 -O2 -arch=sm_60 -Xcompiler -fPIC -I./inc -c src/kernels/vertex_centric.cu -o obj/kernels/vertex_centric.o
if errorlevel 1 (
    echo ERROR: Failed to compile vertex-centric kernel.
    pause
    exit /b 1
)

nvcc -std=c++17 -O2 -arch=sm_60 -Xcompiler -fPIC -I./inc -c src/kernels/edge_centric.cu -o obj/kernels/edge_centric.o
if errorlevel 1 (
    echo ERROR: Failed to compile edge-centric kernel.
    pause
    exit /b 1
)

REM Create static library
echo Creating static library...
ar rcs lib/libcraph.a obj/sparse_structs.o obj/kernels/vertex_centric.o obj/kernels/edge_centric.o
if errorlevel 1 (
    echo ERROR: Failed to create static library.
    pause
    exit /b 1
)

REM Build tests
echo Building tests...
g++ -std=c++17 -Wall -Wextra -I./inc tests/simple_test.cpp lib/libcraph.a -o tests/simple_test.exe
if errorlevel 1 (
    echo ERROR: Failed to build simple test.
    pause
    exit /b 1
)

g++ -std=c++17 -Wall -Wextra -I./inc tests/test_sparse_formats.cpp lib/libcraph.a -o tests/comprehensive_test.exe
if errorlevel 1 (
    echo ERROR: Failed to build comprehensive test.
    pause
    exit /b 1
)

nvcc -std=c++17 -O2 -arch=sm_60 -I./inc tests/cuda_test.cu lib/libcraph.a -o tests/cuda_test.exe
if errorlevel 1 (
    echo ERROR: Failed to build CUDA test.
    pause
    exit /b 1
)

echo.
echo Build completed successfully!
echo.
echo Available executables:
echo   tests/simple_test.exe
echo   tests/comprehensive_test.exe
echo   tests/cuda_test.exe
echo.
echo To run tests:
echo   tests/simple_test.exe
echo   tests/comprehensive_test.exe
echo   tests/cuda_test.exe
echo.
pause 