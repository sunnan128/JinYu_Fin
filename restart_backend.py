#!/usr/bin/env python3
"""
金语AI 后端自恢复脚本
功能：检测端口占用 → 清理所有占用进程 → 等待端口彻底释放 → 启动后端 → 验证就绪

被前端「重连」按钮调用，也可独立运行。

用法：python restart_backend.py
"""
import os
import sys
import time
import socket
import re
import subprocess
import urllib.request

PORT = 8006
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def _parse_netstat_port(port: int) -> list:
    """解析 netstat -ano 输出，精确匹配指定端口的 PID 列表

    用 Python 逐行解析替代 findstr 子串匹配，避免将 :80060、:80061 等误匹配为 :8006。
    netstat 输出格式示例：
      TCP    0.0.0.0:8006           0.0.0.0:0              LISTENING      12345
      TCP    [::]:8006              [::]:0                 LISTENING      12345
      TCP    127.0.0.1:8006         127.0.0.1:54321        TIME_WAIT      0
    """
    try:
        result = subprocess.run(
            ['netstat', '-ano'],
            capture_output=True, text=True, timeout=10
        )
        pids = []
        port_str = f":{port}"
        pat = re.compile(rf':({port})\s+')

        for line in result.stdout.splitlines():
            if port_str not in line:
                continue
            # 确认端口号精确匹配（不是 :80060 等子串匹配）
            m = pat.search(line)
            if not m:
                continue
            parts = [p for p in line.split() if p]
            if parts:
                pid_str = parts[-1]
                if pid_str.isdigit():
                    pid = int(pid_str)
                    if pid > 0:  # PID=0 是系统空闲进程，不处理
                        pids.append(pid)
        return sorted(set(pids))
    except Exception:
        return []


def _check_port_socket(host: str, port: int) -> bool:
    """用 socket.bind() 精确检测端口是否真正可用

    不设 SO_REUSEADDR，只有端口完全空闲时 bind 才会成功。
    这是最权威的检测方式——能被 bind 说明端口完全空闲。
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        s.close()
        return True
    except OSError:
        pass
    finally:
        try:
            s.close()
        except Exception:
            pass
    return False


def find_listening_processes(port: int) -> list:
    """仅查找 LISTENING 状态的进程（端口实际占用者）"""
    pids = _parse_netstat_port(port)
    if not pids:
        return []
    try:
        result = subprocess.run(
            f'netstat -ano | findstr "{port}" | findstr "LISTENING"',
            shell=True, capture_output=True, text=True, timeout=10
        )
        listening_pids = set()
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            parts = [p for p in line.split() if p]
            if parts:
                pid_str = parts[-1]
                if pid_str.isdigit():
                    listening_pids.add(int(pid_str))
        # 只返回也在 _parse_netstat_port 结果中的 PID（精确匹配）
        return sorted(pid for pid in listening_pids if pid in pids)
    except Exception:
        return []


def kill_processes(pids: list) -> None:
    """强制终止进程列表（Windows 用 taskkill /F）"""
    if not pids:
        return
    for pid in pids:
        try:
            result = subprocess.run(
                f'taskkill /F /PID {pid}',
                shell=True, capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                print(f"  ✅ 已终止 PID {pid}")
            elif "not found" in result.stderr.lower():
                print(f"  PID {pid} 已不存在（自动消失）")
            else:
                print(f"  ⚠ 终止 PID {pid} 失败: {result.stderr.strip()}")
        except Exception as e:
            print(f"  ⚠ 终止 PID {pid} 异常: {e}")


def wait_port_free(port: int, max_wait: int = 15) -> bool:
    """等待端口释放，最多 max_wait 秒。

    使用 socket.bind() 作为权威检测方式（比 netstat 更可靠）。
    杀死进程后端口不会立即释放（Windows TCP TIME_WAIT 约 2-4 秒），
    需要持续轮询直到确认可用。

    Returns:
        True  端口已释放（socket.bind() 成功）
        False 超时后端口仍不可用
    """
    start = time.time()
    for i in range(max_wait):
        if _check_port_socket('0.0.0.0', port):
            elapsed = time.time() - start
            print(f"  端口 {port} 已释放（耗时 {elapsed:.1f} 秒）")
            return True
        time.sleep(1)
    return False


def restart_backend() -> bool:
    """重启后端服务
    返回 True 表示启动成功，False 表示失败
    """
    # ── 第 0 步：先检查端口是否本来就空闲 ──
    if _check_port_socket('0.0.0.0', PORT):
        print(f"[0/5] 端口 {PORT} 当前空闲，跳过清理")
        go_direct = True
    else:
        go_direct = False

    if not go_direct:
        print(f"[1/5] 检测端口 {PORT} 占用情况...")

        # 第一步：查所有关联进程
        all_pids = _parse_netstat_port(PORT)
        if all_pids:
            print(f"  发现 {len(all_pids)} 个关联进程: {all_pids}")
            kill_processes(all_pids)
        else:
            print("  未找到关联进程，端口可能被非 Python 进程占用")

        # 第二步：通过名称搜索可能的 Python uvicorn 进程（兜底）
        try:
            result = subprocess.run(
                'tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH',
                shell=True, capture_output=True, text=True, timeout=5
            )
        except Exception:
            result = None

        # 第三步：等待端口彻底释放
        print(f"\n[2/5] 等待端口 {PORT} 释放...")
        if not wait_port_free(PORT, max_wait=15):
            remaining = _parse_netstat_port(PORT)
            print(f"  ❌ 端口 {PORT} 在 15 秒内未释放")
            if remaining:
                print(f"  剩余进程: {remaining}")
                print(f"  💡 尝试强制终止: taskkill /F /PID {' /PID '.join(map(str, remaining))}")
            else:
                print(f"  无关联进程，可能是系统保留或非 Python 进程占用")
            return False
        print("  端口已就绪")

    # 第四步：启动后端
    print(f"\n[{'3' if not go_direct else '1'}/5] 启动后端 (端口 {PORT})...")
    backend_dir = os.path.join(PROJECT_DIR, "backend")
    if not os.path.exists(backend_dir):
        print(f"  ❌ 后端目录不存在: {backend_dir}")
        return False

    python_exe = sys.executable
    try:
        subprocess.Popen(
            [
                python_exe, '-m', 'uvicorn', 'backend.main:app',
                '--host', '0.0.0.0', '--port', str(PORT)
            ],
            cwd=PROJECT_DIR,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        print(f"  启动命令已发送")
        print(f"  Python: {python_exe}")
        print(f"  端口:   {PORT}")
    except Exception as e:
        print(f"  ❌ 启动失败: {e}")
        return False

    # 验证就绪
    print(f"\n[{'4' if not go_direct else '2'}/5] 等待服务就绪...")
    for i in range(30):
        try:
            resp = urllib.request.urlopen(f'http://localhost:{PORT}/health', timeout=2)
            if resp.status == 200:
                print(f"  ✅ 后端服务已就绪 (耗时 {i+1} 秒)")
                return True
        except Exception:
            pass
        time.sleep(1)

    print(f"  ⚠ 后端服务启动超时，请检查新窗口中是否有报错")
    return False


if __name__ == "__main__":
    print("=" * 50)
    print("    金语AI 后端自恢复脚本")
    print("=" * 50)
    print(f"  工作目录: {PROJECT_DIR}")
    print(f"  端口:     {PORT}")
    print(f"  时间:     {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    success = restart_backend()
    print()
    print("=" * 50)
    if success:
        print("  结果: ✅ 服务已恢复")
    else:
        print("  结果: ❌ 启动失败，请查看上方日志诊断")
    print("=" * 50)
    sys.exit(0 if success else 1)
