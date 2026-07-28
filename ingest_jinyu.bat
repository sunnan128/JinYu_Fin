@echo off
cd /d %~dp0

REM ===== 与 start_jinyu.bat 共用同一套虚拟环境 =====
set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
    echo [ERROR] .venv 不存在: %~dp0.venv\Scripts\python.exe
    echo          请先在项目根目录创建虚拟环境并安装依赖。
    pause
    exit /b 1
)

echo ================================================
echo     JinYu 一键灌库 (Ingest Raw Documents)
echo ================================================
echo.

REM ---- 命令行直接传参时，跳过菜单 ----
if "%1"=="dry"   goto DRY
if "%1"=="run"   goto RUN
if "%1"=="force" goto FORCE
if not "%1"=="" (
    echo [ERROR] 未知参数: %1
    echo 用法: ingest_jinyu.bat [dry ^| run ^| force]
    echo   dry   预览（不写库）   run   正常灌库（去重跳过）   force 强制全量重灌
    pause
    exit /b 1
)

: MENU
echo 请选择操作（输入数字后回车）:
echo   1) 预览 dry-run   —— 只列出将灌库的文件与片段数，不写入向量库（推荐先做）
echo   2) 灌库 run       —— 按文件名去重，库里已有的自动跳过
echo   3) 强制重灌 force —— 全量覆盖（用于元数据改了要刷新）
echo   4) 退出
set /p CHOICE=请输入 [1-4]: 
if "%CHOICE%"=="1" goto DRY
if "%CHOICE%"=="2" goto RUN
if "%CHOICE%"=="3" goto FORCE
if "%CHOICE%"=="4" goto END
echo 无效输入，请重新选择。
goto MENU

: DRY
echo.
echo [1/1] 预览模式：仅扫描，不写库...
echo --------------------------------------------------
"%PY%" ingest_raw.py --dry-run
goto END

: RUN
echo.
echo [1/1] 灌库模式：按文件名去重，已存在则跳过...
echo --------------------------------------------------
"%PY%" ingest_raw.py
goto END

: FORCE
echo.
echo [1/1] 强制重灌：全量覆盖（含已存在文档）...
echo --------------------------------------------------
"%PY%" ingest_raw.py --force
goto END

: END
echo.
echo ================================================
echo    操作结束。可在上方日志查看结果。
echo ================================================
echo.
pause
