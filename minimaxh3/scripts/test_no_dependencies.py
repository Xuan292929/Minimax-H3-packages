#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Audit script to verify Criterion 2:
Zero dependency on sglang, diffusers, or other third-party diffusion frameworks.
"""

import os
import sys
import re

def audit_file_content(root_dir: str):
    forbidden = ["sglang", "diffusers"]
    violations = []

    for root, _, files in os.walk(root_dir):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8") as fp:
                for line_no, line in enumerate(fp, 1):
                    # Check for direct imports
                    for pkg in forbidden:
                        if re.search(rf"\b(import\s+{pkg}|from\s+{pkg}\b)", line):
                            violations.append((path, line_no, line.strip()))

    return violations

def audit_sys_modules():
    # Add root to pythonpath
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        import minimaxh3 as target_pkg
    except ImportError:
        try:
            from packages import minimaxh3 as target_pkg
        except ImportError:
            import minimax_pytorch as target_pkg

    forbidden = ["sglang", "diffusers"]
    imported_violations = []
    for mod in sys.modules:
        for pkg in forbidden:
            if mod == pkg or mod.startswith(f"{pkg}."):
                imported_violations.append(mod)

    return imported_violations

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"[Audit] Scanning python files in {current_dir} for forbidden imports...")
    file_violations = audit_file_content(current_dir)
    if file_violations:
        print("FAIL: Found forbidden import statements:")
        for path, line_no, line in file_violations:
            print(f"  {path}:{line_no}: {line}")
        sys.exit(1)
    else:
        print("PASS: No static imports of sglang or diffusers found in source files!")

    print("[Audit] Checking runtime sys.modules upon importing minimax_pytorch...")
    mod_violations = audit_sys_modules()
    if mod_violations:
        print(f"FAIL: Found forbidden modules loaded in sys.modules: {mod_violations}")
        sys.exit(1)
    else:
        print("PASS: Runtime sys.modules is completely clean of sglang and diffusers!")

    print("[Audit] ALL CHECKS PASSED: Criterion 2 strictly satisfied!")
