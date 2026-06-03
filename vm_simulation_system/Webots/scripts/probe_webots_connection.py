"""Probe whether Webots accepts an extern controller (writes %TEMP%\\webots_probe.log)."""
import os
import sys
import tempfile
from pathlib import Path

LOG = Path(tempfile.gettempdir()) / "webots_probe.log"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def _add_dll_dirs(webots_home: str) -> None:
    if os.name != "nt" or sys.version_info < (3, 8):
        return
    for sub in (
        os.path.join("lib", "controller"),
        os.path.join("msys64", "mingw64", "bin", "cpp"),
        os.path.join("msys64", "mingw64", "bin"),
    ):
        path = os.path.join(webots_home, sub)
        if os.path.isdir(path):
            try:
                os.add_dll_directory(path)
            except (AttributeError, OSError):
                pass


def main() -> int:
    if LOG.exists():
        LOG.unlink()

    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    log(f"python={sys.executable} ({ver})")
    log(f"WEBOTS_HOME={os.environ.get('WEBOTS_HOME', '(unset)')}")
    log(f"WEBOTS_ROBOT_NAME={os.environ.get('WEBOTS_ROBOT_NAME', '(unset)')}")
    log(f"WEBOTS_PID={os.environ.get('WEBOTS_PID', '(unset)')}")

    webots_home = os.environ.get("WEBOTS_HOME")
    if not webots_home:
        log("ERR: set WEBOTS_HOME to your Webots install folder")
        return 2

    api_folder = f"python{sys.version_info.major}{sys.version_info.minor}"
    lib = os.path.join(webots_home, "lib", "controller", api_folder)
    if not os.path.isdir(lib):
        log(f"ERR: missing {lib}")
        log(f"     Webots API must match Python {ver} (do not use python38 with 3.9 venv)")
        return 2

    for sub in (
        os.path.join("lib", "controller"),
        os.path.join("msys64", "mingw64", "bin"),
        os.path.join("msys64", "mingw64", "bin", "cpp"),
    ):
        p = os.path.join(webots_home, sub)
        if os.path.isdir(p):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

    sys.path = [p for p in sys.path if "lib\\controller\\python" not in p.replace("/", "\\")]
    sys.path.insert(0, lib)
    log(f"using API={lib}")
    _add_dll_dirs(webots_home)

    log("importing controller...")
    try:
        from controller import Supervisor
    except Exception as exc:
        log(f"IMPORT ERR: {exc}")
        return 2

    log("calling Supervisor()...")
    try:
        sup = Supervisor()
        step = int(sup.getBasicTimeStep())
        name = sup.getName()
        log(f"OK robot={name} timestep={step}")
        del sup
        return 0
    except Exception as exc:
        log(f"Supervisor ERR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
