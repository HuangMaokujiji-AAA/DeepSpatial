#!/usr/bin/env bash
# 在 Python 启动前优先加载当前环境的 nvJitLink；不更改系统 CUDA 或安装包。
set -euo pipefail

python_bin="${PYTHON:-python}"
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# 仅调用标准库查询位置，不能先 import torch，否则会在设置路径前失败。
nvjitlink_dir="$("$python_bin" -c 'import sysconfig; from pathlib import Path; print(Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "nvjitlink" / "lib")')"
if [[ -f "$nvjitlink_dir/libnvJitLink.so.12" ]]; then
    export LD_LIBRARY_PATH="$nvjitlink_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# 无参数时运行测试；传入参数时可启动训练或执行 python -c 等检查。
if [[ $# -eq 0 ]]; then
    set -- "$project_dir/tests/test_pipeline.py"
fi
exec "$python_bin" "$@"
