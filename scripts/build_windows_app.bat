@echo off
REM Simple script to build the GetDist GUI Windows executable
REM This script should be run on Windows

REM Check if running on Windows
if not "%OS%"=="Windows_NT" (
    echo Error: This script must be run on Windows
    exit /b 1
)

REM Get the script directory and repository root
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."

REM Set up variables
set "OUTPUT_DIR=%REPO_ROOT%\dist"
set "PROJECT_DIR=%REPO_ROOT%\build_env"

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :end_parse_args
if "%~1"=="--output-dir" (
    set "OUTPUT_DIR=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--project-dir" (
    set "PROJECT_DIR=%~2"
    shift
    shift
    goto :parse_args
)
echo Unknown parameter: %~1
exit /b 1
:end_parse_args

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Check if uv is installed
where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Installing uv package manager...
    curl -LsSf https://astral.sh/uv/install.sh | sh
    REM Add uv to PATH for this session - on Windows it installs to .local/bin
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

REM Verify uv is working
echo Verifying uv installation...
uv --version

REM Clean up any existing environment
if exist "%PROJECT_DIR%" (
    echo Removing existing environment at %PROJECT_DIR%
    rmdir /s /q "%PROJECT_DIR%"
)

REM Verify the icon exists
if not exist "%REPO_ROOT%\getdist\gui\images\Icon.ico" (
    echo Warning: Icon.ico not found, will use Icon.png as fallback
)

REM Verify the main script exists
if not exist "%REPO_ROOT%\getdist\gui\mainwindow.py" (
    echo ERROR: mainwindow.py not found at %REPO_ROOT%\getdist\gui\mainwindow.py
    echo Listing files in the repository:
    dir /s /b "%REPO_ROOT%\getdist\gui\*.py"
    exit /b 1
) else (
    echo Found mainwindow.py at %REPO_ROOT%\getdist\gui\mainwindow.py
)

REM Build the app
echo Building Windows executable...
cd "%REPO_ROOT%"
python "%SCRIPT_DIR%build_windows_app.py" --output-dir "%OUTPUT_DIR%" --project-dir "%PROJECT_DIR%"

if %ERRORLEVEL% neq 0 (
    echo Error: App build failed
    exit /b 1
)

echo Build completed successfully!
exit /b 0
