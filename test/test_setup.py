import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_IS_WINDOWS = sys.platform == "win32"
_has_pytest_timeout = importlib.util.find_spec("pytest_timeout") is not None

# The install tests shell out to full pip/uv builds and legitimately run for
# minutes (243s observed in CI, longer on a loaded runner), so a suite-wide
# --timeout=120 is too tight for them. The marker is applied only when
# pytest-timeout is installed: the minimal setup-test job runs this file
# without the plugin.
pytestmark = [pytest.mark.timeout(600)] if _has_pytest_timeout else []


def _run(cmd, *, cwd, env=None, timeout=60 * 60) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed.\n"
            f"cwd: {cwd}\n"
            f"cmd: {cmd}\n"
            f"exit_code: {proc.returncode}\n"
            f"output:\n{proc.stdout}"
        )
    return proc.stdout


def _pip_uninstall(pkg: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", pkg],
        cwd=str(_ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _install_cmd_prefix(installer: str) -> list[str]:
    if installer == "uv":
        return ["uv", "pip", "install", "--python", sys.executable]
    return [sys.executable, "-m", "pip", "install"]


def _git(args) -> str:
    return _run(["git", *args], cwd=_ROOT, timeout=60).strip()


def _expected_dist_version(base_version: str) -> str:
    # Match the same release-branch rule as setup.py.
    branch = None
    for key in ("GITHUB_REF_NAME", "GIT_BRANCH", "BRANCH_NAME", "CI_COMMIT_REF_NAME"):
        val = os.environ.get(key)
        if val:
            branch = val
            break
    if branch is None:
        b = _git(["rev-parse", "--abbrev-ref", "HEAD"])
        branch = None if b == "HEAD" else b
    if branch is not None and branch.startswith("refs/heads/"):
        branch = branch[len("refs/heads/") :]

    if branch is not None and (
        branch == f"release/v{base_version}"
        or branch.endswith(f"/release/v{base_version}")
    ):
        return base_version

    return f"{base_version}+g{_git(['rev-parse', '--short', 'HEAD'])}"


def test_build_extension_owns_cxx_standard_or_uses_alias():
    spec = importlib.util.spec_from_file_location("torchrl_setup", _ROOT / "setup.py")
    setup_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup_module)

    compatibility_flag = setup_module._get_cxx20_compatibility_flag()
    for extension in setup_module.get_extensions():
        for compiler, flags in extension.extra_compile_args.items():
            standard_flags = [
                flag for flag in flags if "std=" in flag or "std:" in flag
            ]
            expected = (
                [compatibility_flag]
                if compiler == "cxx" and compatibility_flag is not None
                else []
            )
            assert standard_flags == expected


def test_detect_torch_extension_cxx20(monkeypatch):
    spec = importlib.util.spec_from_file_location("torchrl_setup", _ROOT / "setup.py")
    setup_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup_module)

    monkeypatch.setattr(
        setup_module.inspect,
        "getsource",
        lambda _: "cpp_flag = cpp_flag_prefix + 'c++20'",
    )

    assert setup_module._torch_extension_requires_cxx20()


def test_legacy_cxx20_spelling_fallback(monkeypatch):
    spec = importlib.util.spec_from_file_location("torchrl_setup", _ROOT / "setup.py")
    setup_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup_module)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0 if "-std=c++2a" in command else 1)

    monkeypatch.setattr(setup_module.subprocess, "run", run)
    monkeypatch.setattr(setup_module.sys, "platform", "linux")
    monkeypatch.setattr(setup_module, "_torch_extension_requires_cxx20", lambda: True)

    assert setup_module._get_cxx20_compatibility_flag() == "-std=c++2a"
    assert [command[-5] for command in calls] == ["-std=c++20", "-std=c++2a"]


def test_no_cxx20_fallback_for_older_torch(monkeypatch):
    spec = importlib.util.spec_from_file_location("torchrl_setup", _ROOT / "setup.py")
    setup_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup_module)

    monkeypatch.setattr(setup_module, "_torch_extension_requires_cxx20", lambda: False)

    def run(*args, **kwargs):
        raise AssertionError("The compiler should not be probed for a C++17 build")

    monkeypatch.setattr(setup_module.subprocess, "run", run)

    assert setup_module._get_cxx20_compatibility_flag() is None


def _make_rocm_home(root: Path, layout: str | None) -> str | None:
    """Lay out the ROCm files that decide whether hipcc can compile device code."""
    if layout is None:
        return None
    files = {
        # Device libraries where clang looks for them: in <ROCm>/amdgcn/bitcode
        # or in its resource directory.
        "rocm": ["bin/hipcc", "amdgcn/bitcode/ocml.bc"],
        "rocm-clang-resource-dir": [
            "bin/hipcc",
            "lib/llvm/lib/clang/22/lib/amdgcn/bitcode/ocml.bc",
        ],
        # hipcc present, device libraries in a directory clang does not search
        # (the core-only ROCm SDK wheel layout).
        "rocm-no-device-libs": ["bin/hipcc", "lib/llvm/amdgcn/bitcode/ocml.bc"],
        # ROCm runtime without a compiler.
        "rocm-no-hipcc": ["lib/libamdhip64.so", "amdgcn/bitcode/ocml.bc"],
    }[layout]
    for name in files:
        path = root / name
        if name == "bin/hipcc" and _IS_WINDOWS:
            path = path.with_name("hipcc.exe")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return str(root)


@pytest.mark.parametrize(
    "hip_version,cuda_version,rocm_layout,nvcc_version,force_cuda,expected",
    [
        pytest.param("7.2", None, "rocm", None, "0", True, id="rocm"),
        pytest.param(
            "7.2",
            None,
            "rocm-clang-resource-dir",
            None,
            "0",
            True,
            id="rocm-clang-resource-dir",
        ),
        pytest.param(
            "7.2",
            None,
            "rocm-no-device-libs",
            None,
            "0",
            False,
            id="rocm-no-device-libs",
        ),
        pytest.param(
            "7.2", None, "rocm-no-hipcc", None, "0", False, id="rocm-no-hipcc"
        ),
        pytest.param(
            "7.2",
            None,
            "rocm-no-device-libs",
            None,
            "1",
            True,
            id="rocm-no-device-libs-force-cuda",
        ),
        pytest.param("7.2", None, None, None, "0", False, id="rocm-no-rocm-home"),
        pytest.param("7.2", None, None, None, "1", True, id="rocm-force-cuda"),
        pytest.param(None, "12.8", None, "12.8", "0", True, id="cuda"),
        pytest.param(None, "12.8", None, "12.6", "0", False, id="cuda-nvcc-mismatch"),
        pytest.param(None, None, "rocm", None, "0", False, id="cpu"),
    ],
)
def test_should_build_gpu_extension(
    monkeypatch,
    tmp_path: Path,
    hip_version,
    cuda_version,
    rocm_layout,
    nvcc_version,
    force_cuda,
    expected,
):
    spec = importlib.util.spec_from_file_location("torchrl_setup", _ROOT / "setup.py")
    setup_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup_module)

    monkeypatch.setenv("FORCE_CUDA", force_cuda)
    monkeypatch.delenv("HIP_DEVICE_LIB_PATH", raising=False)
    monkeypatch.setattr(setup_module.torch.version, "hip", hip_version)
    monkeypatch.setattr(setup_module.torch.version, "cuda", cuda_version)
    monkeypatch.setattr(
        setup_module, "ROCM_HOME", _make_rocm_home(tmp_path, rocm_layout)
    )
    monkeypatch.setattr(
        setup_module, "CUDA_HOME", "/usr/local/cuda" if nvcc_version else None
    )
    monkeypatch.setattr(setup_module, "_get_nvcc_cuda_version", lambda: nvcc_version)

    assert setup_module._should_build_cuda() is expected


def test_rocm_without_device_libs_warns(monkeypatch, tmp_path: Path, caplog):
    spec = importlib.util.spec_from_file_location("torchrl_setup", _ROOT / "setup.py")
    setup_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup_module)

    monkeypatch.setenv("FORCE_CUDA", "0")
    monkeypatch.delenv("HIP_DEVICE_LIB_PATH", raising=False)
    monkeypatch.setattr(setup_module.torch.version, "hip", "7.2")
    monkeypatch.setattr(setup_module.torch.version, "cuda", None)
    rocm_home = _make_rocm_home(tmp_path, "rocm-no-device-libs")
    monkeypatch.setattr(setup_module, "ROCM_HOME", rocm_home)

    with caplog.at_level("WARNING"):
        assert setup_module._should_build_cuda() is False
    assert "Set FORCE_CUDA=1" in caplog.text
    assert rocm_home in caplog.text

    # An explicit device library path is what clang uses instead.
    monkeypatch.setenv("HIP_DEVICE_LIB_PATH", str(tmp_path / "lib/llvm/amdgcn/bitcode"))
    assert setup_module._should_build_cuda() is True


@pytest.mark.skipif(_IS_WINDOWS, reason="checks the Unix compiler flags")
@pytest.mark.parametrize(
    "hip_version,debug_flag", [("7.2", "-g"), (None, "-G")], ids=["rocm", "cuda"]
)
def test_debug_build_device_debug_flag(monkeypatch, hip_version, debug_flag):
    spec = importlib.util.spec_from_file_location("torchrl_setup", _ROOT / "setup.py")
    setup_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup_module)

    monkeypatch.setenv("DEBUG", "1")
    monkeypatch.setattr(setup_module.torch.version, "hip", hip_version)
    monkeypatch.setattr(setup_module, "_should_build_cuda", lambda: True)
    monkeypatch.setattr(
        setup_module, "CUDAExtension", lambda name, sources, **kwargs: kwargs
    )

    (extension,) = setup_module.get_extensions()
    assert extension["extra_compile_args"]["nvcc"] == ["-O0", debug_flag, "-DWITH_CUDA"]


def test_hip_version_change_cleans_stale_extension(monkeypatch, tmp_path: Path):
    spec = importlib.util.spec_from_file_location("torchrl_setup", _ROOT / "setup.py")
    setup_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup_module)

    monkeypatch.setenv("FORCE_CUDA", "0")
    monkeypatch.setattr(setup_module.torch.version, "hip", "7.2")
    monkeypatch.setattr(setup_module.torch.version, "cuda", None)
    monkeypatch.setattr(
        setup_module, "ROCM_HOME", _make_rocm_home(tmp_path / "rocm", "rocm")
    )
    monkeypatch.setattr(setup_module, "ROOT_DIR", tmp_path)
    build_info_file = tmp_path / "build" / ".torchrl_build_info.json"
    monkeypatch.setattr(setup_module, "_BUILD_INFO_FILE", build_info_file)

    build_info_file.parent.mkdir()
    build_info_file.write_text(
        json.dumps(
            {
                "torch_version": setup_module.torch.__version__,
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "cuda_version": None,
                "hip_version": "7.1",
                "build_cuda_version": None,
                "force_cuda": False,
                "build_cuda": True,
            }
        )
    )
    ext = ".pyd" if _IS_WINDOWS else ".so"
    stale = (
        tmp_path
        / "torchrl"
        / f"_torchrl.cpython-{sys.version_info.major}{sys.version_info.minor}-x{ext}"
    )
    stale.parent.mkdir()
    stale.touch()

    setup_module._check_and_clean_stale_builds()

    assert not stale.exists()
    assert json.loads(build_info_file.read_text())["hip_version"] == "7.2"


@pytest.mark.parametrize(
    "editable",
    [
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                _IS_WINDOWS, reason="Windows file locking prevents editable re-install"
            ),
            id="editable",
        ),
        pytest.param(False, id="wheel"),
    ],
)
def test_install_no_deps_has_nonzero_version(editable: bool, tmp_path: Path):
    # Requires git checkout.
    if not (_ROOT / ".git").exists():
        pytest.skip("not a git checkout")

    base_version = (_ROOT / "version.txt").read_text().strip()
    if not base_version:
        raise RuntimeError("Empty version.txt")

    expected = _expected_dist_version(base_version)

    installer = "uv" if shutil.which("uv") is not None else "pip"

    # Ensure we cover the historical failure mode where version becomes 0.0.0 when
    # build requirements aren't present (e.g. --no-build-isolation). This dedicated
    # CI job intentionally runs with --no-deps installs.
    _pip_uninstall("setuptools_scm")

    # Ensure clean re-install for each case.
    _pip_uninstall("torchrl")

    cmd = _install_cmd_prefix(installer)
    cmd.append("--no-deps")
    cmd.append("--no-build-isolation")
    if editable:
        cmd.extend(["-e", "."])
    else:
        cmd.append(".")

    _run(cmd, cwd=_ROOT, timeout=60 * 60)

    probe_dir = tmp_path / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)

    code = r"""
import importlib.metadata as md
import json

out = {}
out["dist_version"] = md.version("torchrl")
try:
    import torchrl
    out["pkg_version"] = getattr(torchrl, "__version__", None)
    out["pkg_file"] = getattr(torchrl, "__file__", None)
except Exception as err:
    out["import_error"] = repr(err)

print(json.dumps(out))
"""
    out = _run([sys.executable, "-c", code], cwd=probe_dir, timeout=5 * 60)
    # Warnings or deprecation notices may be printed to stdout (stderr is
    # merged), so extract only the JSON object which is always the last line.
    json_line = ""
    for line in reversed(out.strip().splitlines()):
        if line.strip().startswith("{"):
            json_line = line.strip()
            break
    if not json_line:
        raise RuntimeError(
            f"Probe script produced no JSON output.\nFull output:\n{out}"
        )
    info = json.loads(json_line)

    dist_version = str(info["dist_version"]).strip()
    assert dist_version != "0.0.0"
    assert dist_version == expected

    pkg_version = info.get("pkg_version")
    pkg_file = info.get("pkg_file")
    if pkg_version is not None and pkg_file is not None:
        pkg_version = str(pkg_version).strip()
        assert pkg_version != "0.0.0"
        assert pkg_version == expected

        pkg_path = Path(pkg_file).resolve()
        if editable:
            assert str(pkg_path).startswith(str(_ROOT.resolve()))
        else:
            assert "site-packages" in str(pkg_path)
    else:
        # If some hard dependency is missing, import can fail. The packaging version
        # should still be correct.
        assert "dist_version" in info
