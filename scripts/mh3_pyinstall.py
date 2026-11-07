#!/usr/bin/env python3
"""不经过 uv/pip 安装 Python 包：只创建和写入文件，不做改名。

平台的个人 AFS 和会话临时目录上，uv 与 setuptools 安装时的改名会报
"Invalid cross-device link"，所以 venv 里的包都用这个脚本安装。
必须用目标 venv 的 python 运行，安装位置取自它的 sysconfig。

用法：
  python mh3_pyinstall.py wheel <whl>...               安装 wheel
  python mh3_pyinstall.py tree <包目录> <发行名> <版本>  复制纯 Python 包并写 dist-info
  python mh3_pyinstall.py pth <名称> <目录> <版本>      写 .pth，让源码目录可导入
"""

import configparser
import os
import re
import shutil
import sys
import sysconfig
import zipfile

PATHS = sysconfig.get_paths()
SITE = PATHS["purelib"]
PY_TAG = f"python{sys.version_info[0]}.{sys.version_info[1]}"
INSTALLER = b"ss4-pyinstall\n"
CHUNK = 1 << 20
# csrc 是 torchvision 的 C++ 源码，运行时用不到
SKIP_DIRS = {"__pycache__", "csrc"}

LAUNCHER = """#!{python}
import sys
from {module} import {head}
if __name__ == "__main__":
    sys.exit({call}())
"""


def fail(msg):
    print(f"[pyinstall] 错误：{msg}", file=sys.stderr)
    sys.exit(1)


def normalize(name):
    return re.sub(r"[-_.]+", "_", name)


def safe_join(base, rel):
    parts = rel.replace("\\", "/").split("/")
    if rel.startswith("/") or ".." in parts:
        fail(f"非法路径 {rel}")
    return os.path.join(base, *[p for p in parts if p])


def make_executable(path):
    try:
        os.chmod(path, 0o755)
    except OSError as exc:
        print(f"[pyinstall] 警告：无法设置可执行权限 {path}: {exc}")


def write_stream(dest, src, executable=False):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as out:
        shutil.copyfileobj(src, out, CHUNK)
    if executable:
        make_executable(dest)


def write_bytes(dest, data, executable=False):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as out:
        out.write(data)
    if executable:
        make_executable(dest)


def write_dist_info(name, version):
    info = os.path.join(SITE, f"{normalize(name)}-{version}.dist-info")
    meta = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    write_bytes(os.path.join(info, "METADATA"), meta.encode())
    write_bytes(os.path.join(info, "INSTALLER"), INSTALLER)
    write_bytes(os.path.join(info, "RECORD"), b"")


def write_console_scripts(zf, info_dir):
    try:
        text = zf.read(f"{info_dir}/entry_points.txt").decode("utf-8")
    except KeyError:
        return 0
    parser = configparser.ConfigParser(delimiters=("=",), interpolation=None)
    parser.optionxform = str
    parser.read_string(text)
    if not parser.has_section("console_scripts"):
        return 0
    count = 0
    for name, target in parser.items("console_scripts"):
        module, _, attr = target.split("[")[0].strip().partition(":")
        module, attr = module.strip(), attr.strip()
        if not module or not attr:
            continue
        code = LAUNCHER.format(
            python=sys.executable, module=module, head=attr.split(".")[0], call=attr
        )
        write_bytes(safe_join(PATHS["scripts"], name), code.encode(), True)
        count += 1
    return count


def already_installed(marker):
    try:
        with open(marker, "rb") as f:
            return f.read() == INSTALLER
    except OSError:
        return False


def install_wheel(path):
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        wheel_meta = [
            i.filename for i in infos
            if re.fullmatch(r"[^/]+\.dist-info/WHEEL", i.filename)
        ]
        if len(wheel_meta) != 1:
            fail(f"{path}: 找不到唯一的 .dist-info/WHEEL")
        info_dir = wheel_meta[0].split("/")[0]
        dist = info_dir[: -len(".dist-info")]
        data_dir = dist + ".data"
        meta = zf.read(wheel_meta[0]).decode("utf-8", "replace").lower()
        root = PATHS["purelib"] if "root-is-purelib: true" in meta else PATHS["platlib"]
        marker = safe_join(root, f"{info_dir}/INSTALLER")
        if already_installed(marker):
            print(f"[pyinstall] {os.path.basename(path)}: 已安装，跳过")
            return
        count = 0
        for info in infos:
            name = info.filename
            if name.endswith("/"):
                continue
            executable = bool((info.external_attr >> 16) & 0o111)
            if name.startswith(data_dir + "/"):
                scheme, _, rest = name[len(data_dir) + 1:].partition("/")
                if scheme in ("purelib", "platlib"):
                    base = PATHS[scheme]
                elif scheme == "scripts":
                    data = zf.read(info)
                    if data.startswith(b"#!python"):
                        data = b"#!" + sys.executable.encode() + data[len(b"#!python"):]
                    write_bytes(safe_join(PATHS["scripts"], rest), data, True)
                    count += 1
                    continue
                elif scheme == "headers":
                    base = os.path.join(sys.prefix, "include", "site", PY_TAG, dist.split("-")[0])
                elif scheme == "data":
                    base = PATHS["data"]
                else:
                    fail(f"{path}: 未知的 .data 子目录 {scheme}")
                dest = safe_join(base, rest)
            else:
                dest = safe_join(root, name)
            with zf.open(info) as src:
                write_stream(dest, src, executable)
            count += 1
        commands = write_console_scripts(zf, info_dir)
        # 标记最后写：安装中断时不会被下次运行当成已完成
        write_bytes(marker, INSTALLER)
    extra = f"，{commands} 个命令" if commands else ""
    print(f"[pyinstall] {os.path.basename(path)}: {count} 个文件{extra}")


def install_tree(src, name, version):
    src = os.path.abspath(src)
    if not os.path.isdir(src):
        fail(f"目录不存在 {src}")
    parent = os.path.dirname(src)
    count = 0
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            if filename.endswith((".pyc", ".pyo")):
                continue
            path = os.path.join(dirpath, filename)
            dest = os.path.join(SITE, os.path.relpath(path, parent))
            with open(path, "rb") as fsrc:
                write_stream(dest, fsrc, os.access(path, os.X_OK))
            count += 1
    write_dist_info(name, version)
    print(f"[pyinstall] {name} {version}: 复制 {count} 个文件")


def install_pth(name, directory, version):
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        fail(f"目录不存在 {directory}")
    write_bytes(os.path.join(SITE, f"__editable__ss4_{name}.pth"), (directory + "\n").encode())
    write_dist_info(name, version)
    print(f"[pyinstall] {name}: .pth -> {directory}")


def main(argv):
    if sys.prefix == sys.base_prefix:
        fail("请用目标 venv 的 python 运行")
    if len(argv) >= 2 and argv[0] == "wheel":
        for path in argv[1:]:
            install_wheel(path)
    elif len(argv) == 4 and argv[0] == "tree":
        install_tree(*argv[1:])
    elif len(argv) == 4 and argv[0] == "pth":
        install_pth(*argv[1:])
    else:
        print(__doc__, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
