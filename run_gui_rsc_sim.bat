@echo off
setlocal
pushd "%~dp0" || exit /b 1
call conda run --no-capture-output -n scripts python rb85_rsc_sim.py --gui
set "rc=%errorlevel%"
popd
exit /b %rc%