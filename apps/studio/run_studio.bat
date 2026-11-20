@echo off
setlocal
rem OpenFAWE Studio (decision 0052): run a case live and watch it in 3D. Double-click.
rem Uses the accelerated worktree's environment (openfawe-accel\local\accel-venv312), like run_platform.bat;
rem the first start installs the viewer packages there (PySide6, pyvista, pyvistaqt, vtk; internet needed once).
rem Runs are stored in %USERPROFILE%\OpenFAWE Studio runs.
chcp 65001 >nul
set "HERE=%~dp0"
for %%I in ("%HERE%..\..") do set "ACCEL=%%~fI"
set "PY=%ACCEL%\local\accel-venv312\Scripts\python.exe"
if not exist "%PY%" (echo [error] %PY% not found: run run_local_benchmark.bat first. & pause & exit /b 1)
"%PY%" -c "import pyvista, pyvistaqt, PySide6" >nul 2>&1
if errorlevel 1 (
  echo Installing the viewer packages into the accelerated environment, once ...
  "%PY%" -m pip install --disable-pip-version-check -r "%ACCEL%\apps\wake_viewer\requirements-viewer.lock"
  if errorlevel 1 (echo [error] Package installation failed; open an issue with the text above. & pause & exit /b 1)
)
set "PYTHONPATH=%ACCEL%\src;%HERE%"
cd /d "%HERE%"
"%PY%" -m openfawe_studio %* & if errorlevel 1 pause
