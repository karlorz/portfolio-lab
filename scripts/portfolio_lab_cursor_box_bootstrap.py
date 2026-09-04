#!/usr/bin/env python3
"""Portfolio Lab cursor-box user-owned dependency bootstrap.

Manages user-owned toolchain artifacts for cursor-box (Alpine Linux x86_64 musl).
Installs strictly under user-owned prefixes without root, apk, sudo, or Docker.
Extracts the pinned official 32-package Alpine v3.22 closure into an isolated
user-owned alpine-root (<prefix>/share/portfolio-lab/toolchain/alpine-root),
extracts the complete standalone Python runtime tree into
(<prefix>/share/portfolio-lab/toolchain/python-root), and creates robust,
relocatable wrappers in <prefix>/bin setting private LD_LIBRARY_PATH,
GIT_EXEC_PATH, GIT_TEMPLATE_DIR, and private CA bundle environment variables.

Usage::

  portfolio-lab-cursor-box-bootstrap.sh dry-run [--prefix PATH]
  portfolio-lab-cursor-box-bootstrap.sh install [--prefix PATH]
  portfolio-lab-cursor-box-bootstrap.sh verify [--prefix PATH]
  portfolio-lab-cursor-box-bootstrap.sh uninstall [--prefix PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "portfolio-lab-cursor-box-bootstrap/v2"
LAYOUT_REVISION = "v2.2-alpine-build-root-bunx"
DEFAULT_PREFIX = "/home/box/.local"
DEFAULT_TOOLCHAIN_SUBDIR = "share/portfolio-lab/toolchain"
DEFAULT_ALPINE_ROOT_SUBDIR = "share/portfolio-lab/toolchain/alpine-root"
DEFAULT_ALPINE_BUILD_ROOT_SUBDIR = "share/portfolio-lab/toolchain/alpine-build-root"
DEFAULT_PYTHON_ROOT_SUBDIR = "share/portfolio-lab/toolchain/python-root"
DEFAULT_STANDALONE_SUBDIR = "share/portfolio-lab/toolchain/standalone"

ALPINE_DL_BASE = "https://dl-cdn.alpinelinux.org/alpine/v3.22/main/x86_64"
ALPINE_COMMUNITY_DL_BASE = "https://dl-cdn.alpinelinux.org/alpine/v3.22/community/x86_64"

# Official 32-package complete self-contained closure for Alpine v3.22 x86_64
ALPINE_PACKAGE_CLOSURE: list[tuple[str, str, str]] = [
    ("acl-libs-2.3.2-r1.apk", "83c35935dcc0d592f65bc4626cec243b2c5f1a0a3d3e1361f10c4bb221a941f1", "LGPL-2.1-or-later AND GPL-2.0-or-later"),
    ("brotli-libs-1.1.0-r2.apk", "a693524543421b3a90f163ccb48d5ad0f5fd773b5c3b640acc461eace2cb01b6", "MIT"),
    ("c-ares-1.34.8-r0.apk", "1397ec9682ff6153e5d037965c76408e570ae6535ce479cc2af37436fdea52ce", "MIT"),
    ("ca-certificates-20260611-r0.apk", "a8ad8f04dfba1a2897388c4b420b698bf1ecd870be10f0127134a567d5e59896", "MPL-2.0 AND MIT"),
    ("ca-certificates-bundle-20260611-r0.apk", "a18fd1bd8bea03966ee5719aa61e44d9a810db2c8b6641b45f92b30e860f0927", "MPL-2.0 AND MIT"),
    ("curl-8.14.1-r3.apk", "ec89d3e01d2507e0ece31746018f0c053a4fbd007628dc3016afab406b5c0ff6", "curl"),
    ("git-2.49.1-r0.apk", "42f4573799ffce0c7dd4954d2247eb882eb87360dc18637930624484ecfd1c90", "GPL-2.0-only"),
    ("jq-1.8.2-r0.apk", "dc73a46f8a2b0827fb6b0590f07d8f456a41f3fae44e34fe7e3a09b4329325e3", "MIT"),
    ("libcrypto3-3.5.8-r0.apk", "1d111bc0ad6380fdda22e6513941dc2e7988d6b1621d535bed6b3fa5ef086fae", "Apache-2.0"),
    ("libcurl-8.14.1-r3.apk", "249e5a3a50558cdb3ce1bab1619b15d88243a628d878700c8d05878c2e71ed71", "curl"),
    ("libexpat-2.8.4-r0.apk", "f20b6ce1da2693c16023de7fe93c90353a4e93f3968ba90ec12729f4f2636864", "MIT"),
    ("libgcc-14.2.0-r6.apk", "04f3467bc967e705221a843fe4d3de5850db826e571686e0c0ed453d38cb5c59", "GPL-2.0-or-later AND LGPL-2.1-or-later"),
    ("libidn2-2.3.7-r0.apk", "515cfe061176c7456ea3548651d0014084d77dd3774b78e2c944404ae75a41ff", "GPL-2.0-or-later OR LGPL-3.0-or-later"),
    ("libncursesw-6.5_p20250503-r0.apk", "aeafdfca68147b014705b4e2564639ade6345198debb522f2c6c51d32e417651", "X11"),
    ("libpsl-0.21.5-r3.apk", "1e79da4fa1364b7153b8950a363a6de0bef1d1b48d46fcb3b850cd4f233727b2", "MIT"),
    ("libssl3-3.5.8-r0.apk", "e8d3ea5e9750cb1f4c4b459d172630925fe6bc13c113c590ac36c78013821e62", "Apache-2.0"),
    ("libstdc++-14.2.0-r6.apk", "939f7c99898f3e8154207a17f4acbe8bc40437e1bb1b43f5525620ca9e452a2e", "GPL-2.0-or-later AND LGPL-2.1-or-later"),
    ("libunistring-1.3-r0.apk", "989640e58f7646c0495c3950df36eb7b4df152d8150c2c888f8ab954fed8c908", "GPL-2.0-or-later OR LGPL-3.0-or-later"),
    ("libxxhash-0.8.3-r0.apk", "e3140cf609508b32db2aec81c8736e7db5d96c351857c99b364a6d1a913bb850", "BSD-2-Clause"),
    ("lz4-libs-1.10.0-r0.apk", "b4f393b9d8ce5a431788b31c899734f84d60a47487a6b833ee8d8012c501fe3f", "BSD-2-Clause AND GPL-2.0-or-later"),
    ("ncurses-terminfo-base-6.5_p20250503-r0.apk", "0815a5f0403974bb9c34d456e71dc9c0222cb5455d393bc63c44b573da3d7fe0", "X11"),
    ("nghttp2-libs-1.69.0-r0.apk", "d6ee515d30d703e94b55ef9c0a02aea053313f654acbd2c655d18e3907ecb66a", "MIT"),
    ("oniguruma-6.9.10-r0.apk", "52f7f6f99840f252d611ded9d72c8ecfdc99888391d6cbc33a7ba6703c271dfe", "BSD-2-Clause"),
    ("pcre2-10.46-r0.apk", "cfb8ad103a101fa6a31769e50e188dab9c60124705682d01b3de268795db58ad", "BSD-3-Clause"),
    ("popt-1.19-r4.apk", "0f00f96b98b63628a5c7d25597dd00d088c2951b31439e415427618ddcee40b5", "MIT"),
    ("readline-8.2.13-r1.apk", "520fa586c689144928191bee13e2c85ff4e170ad87d1471ec48e3e97611673d8", "GPL-3.0-or-later"),
    ("rsync-3.5.0-r0.apk", "696dd5358671f59494fdd8aa49ac4b3b0489e28e5df3d41b80f279b2c1e0d64e", "GPL-3.0-or-later"),
    ("sqlite-3.49.2-r1.apk", "7a2b13ff89bc350b2a04f6977054dbbf48a258eb7c240900028a1ab5b674f010", "blessing"),
    ("sqlite-libs-3.49.2-r1.apk", "7af9a58595642de91ea35d3d070cd38d969680a720ce76b14966859e6fb4af99", "blessing"),
    ("zlib-1.3.2-r0.apk", "1f3d5f463f490dad3a68097376711bfe5e8156e9e8daff3070513aa4378cdeca", "Zlib"),
    ("zstd-1.5.7-r0.apk", "2075231e0d0c123e5a5cede1272ba8adc261728e6edb2736560e73aa0cb41938", "BSD-3-Clause OR GPL-2.0-or-later"),
    ("zstd-libs-1.5.7-r0.apk", "1bdd6e57cfbfbfd6e8481cad37ddd5d199950715bec1879b3afb600272dbb09e", "BSD-3-Clause OR GPL-2.0-or-later"),
]

# Official 58-package complete self-contained build closure for Alpine v3.22 x86_64
# Excludes samurai-1.2-r7.apk in favor of real standalone Ninja.
ALPINE_BUILD_PACKAGE_CLOSURE: list[tuple[str, str, str, str]] = [
    ("acl-libs-2.3.2-r1.apk", "83c35935dcc0d592f65bc4626cec243b2c5f1a0a3d3e1361f10c4bb221a941f1", "LGPL-2.1-or-later AND GPL-2.0-or-later", "main"),
    ("binutils-2.44-r3.apk", "d12005fc43aa4e818ce240dcdd5ae90a42a5a4b3ea6937800c7b0a5b877190aa", "GPL-2.0-or-later AND LGPL-2.1-or-later AND BSD-3-Clause", "main"),
    ("brotli-libs-1.1.0-r2.apk", "a693524543421b3a90f163ccb48d5ad0f5fd773b5c3b640acc461eace2cb01b6", "MIT", "main"),
    ("build-base-0.5-r3.apk", "2788ae90dc82999b26d121536ec2bd1c053e46b470c0202229bd4b4d2347d1b4", "MIT", "main"),
    ("c-ares-1.34.8-r0.apk", "1397ec9682ff6153e5d037965c76408e570ae6535ce479cc2af37436fdea52ce", "MIT", "main"),
    ("ca-certificates-bundle-20260611-r0.apk", "a18fd1bd8bea03966ee5719aa61e44d9a810db2c8b6641b45f92b30e860f0927", "MPL-2.0 AND MIT", "main"),
    ("cargo-1.87.0-r1.apk", "0fc2256e93b120b12069af4116d13b82b6f5878780e810ebe7cfbf1a3b85e8f9", "(Apache-2.0 OR MIT) AND UNLICENSE", "main"),
    ("cmake-3.31.7-r1.apk", "3f99fc6bb19b8e41f33764bf641bb181d57bdd1afb63f2dfee8d4547e1362341", "BSD-3-Clause", "main"),
    ("file-5.46-r2.apk", "94c99dfe7cde72e07aa2b289674e682e821518bfc59e31ce628990d939caed7c", "BSD-2-Clause", "main"),
    ("fortify-headers-1.1-r5.apk", "544a27bd0e002f54677109e2cce594f10f7d0df5321e5e64604e896bba92cdf8", "0BSD", "main"),
    ("g++-14.2.0-r6.apk", "4490ba28b819efdb26a78493e63942605a022d407652eea77027eaba918e6f1a", "GPL-2.0-or-later AND LGPL-2.1-or-later", "main"),
    ("gcc-14.2.0-r6.apk", "4fd10bd05784f596e5ff0adc724a94b71380fac0ac276bd257dd861e8b5822e6", "GPL-2.0-or-later AND LGPL-2.1-or-later", "main"),
    ("gfortran-14.2.0-r6.apk", "b33c727c44716a1afb19aec9dc6a87fc62ee294438fb6537c488bc8941b65f99", "GPL-2.0-or-later AND LGPL-2.1-or-later", "main"),
    ("gmp-6.3.0-r3.apk", "d3f987ae3836ac7774324bff443dd49d03b846209660729d0c30dfff5546e138", "LGPL-3.0-or-later OR GPL-2.0-or-later", "main"),
    ("isl25-0.25-r2.apk", "05bf7a0fe8aad4e1a6cb38ef2dbeac0a2fd05be2bfe862e3d84601b6c3c4f0b6", "MIT", "main"),
    ("jansson-2.14.1-r0.apk", "7fde81421482507163410715a7ef7d7df6a091870edab855c24e8f6fc84e3e2d", "MIT", "main"),
    ("libarchive-3.8.3-r0.apk", "229e3881cbbe148413fd1d04bce78749ec6fe4f36a4c413a8062b7c854c694ce", "BSD-2-Clause AND BSD-3-Clause AND Public-Domain", "main"),
    ("libatomic-14.2.0-r6.apk", "e9a884f8a63f614b722dd5191ede154a6b6b81d99457af610bc6f544a6d2d7bc", "GPL-2.0-or-later AND LGPL-2.1-or-later", "main"),
    ("libbz2-1.0.8-r6.apk", "0c98810f9332bf1bb1b0acc64eaac4b101036689bffba9ea621416b4b3c190cd", "bzip2-1.0.6", "main"),
    ("libcrypto3-3.5.8-r0.apk", "1d111bc0ad6380fdda22e6513941dc2e7988d6b1621d535bed6b3fa5ef086fae", "Apache-2.0", "main"),
    ("libcurl-8.14.1-r3.apk", "249e5a3a50558cdb3ce1bab1619b15d88243a628d878700c8d05878c2e71ed71", "curl", "main"),
    ("libexpat-2.8.4-r0.apk", "f20b6ce1da2693c16023de7fe93c90353a4e93f3968ba90ec12729f4f2636864", "MIT", "main"),
    ("libffi-3.4.8-r0.apk", "9a75cb9024693c1e52c3d8d7c9afb7c79e6e20f6c08df28effdb8dd816095083", "MIT", "main"),
    ("libgcc-14.2.0-r6.apk", "04f3467bc967e705221a843fe4d3de5850db826e571686e0c0ed453d38cb5c59", "GPL-2.0-or-later AND LGPL-2.1-or-later", "main"),
    ("libgfortran-14.2.0-r6.apk", "bb056d2918e8403f177bb103d5897b3e82efaaee3c88435e660cedec42907ce7", "GPL-2.0-or-later AND LGPL-2.1-or-later", "main"),
    ("libgomp-14.2.0-r6.apk", "026e5c9d0bab7e82c42fa11b96ed95e66d2549bd5de5c869c18a3600c2e3aa02", "GPL-2.0-or-later AND LGPL-2.1-or-later", "main"),
    ("libidn2-2.3.7-r0.apk", "515cfe061176c7456ea3548651d0014084d77dd3774b78e2c944404ae75a41ff", "GPL-2.0-or-later OR LGPL-3.0-or-later", "main"),
    ("liblapack-0.3.28-r0.apk", "879ff87556f999d7704e0477ccddb60dc86e66eeb86706658ef3fd67dbb6823c", "BSD-3-Clause", "community"),
    ("liblapacke-0.3.28-r0.apk", "e2e495392ee4a5b29b9fdb8251b457aea49d3bc8b97da6b2d8a00e8412c7632c", "BSD-3-Clause", "community"),
    ("libmagic-5.46-r2.apk", "1fe23639567b42d215d860b2983ed4ccad442c658fbc9082e946823e87c251a8", "BSD-2-Clause", "main"),
    ("libpsl-0.21.5-r3.apk", "1e79da4fa1364b7153b8950a363a6de0bef1d1b48d46fcb3b850cd4f233727b2", "MIT", "main"),
    ("libquadmath-14.2.0-r6.apk", "0274de812a39ec91f1b77736ed40f62e82e10ab5de283c3cb21ddeba549e2ce6", "GPL-2.0-or-later AND LGPL-2.1-or-later", "main"),
    ("libssl3-3.5.8-r0.apk", "e8d3ea5e9750cb1f4c4b459d172630925fe6bc13c113c590ac36c78013821e62", "Apache-2.0", "main"),
    ("libstdc++-14.2.0-r6.apk", "939f7c99898f3e8154207a17f4acbe8bc40437e1bb1b43f5525620ca9e452a2e", "GPL-2.0-or-later AND LGPL-2.1-or-later", "main"),
    ("libstdc++-dev-14.2.0-r6.apk", "13e2169b46a24f03c87caf91387312724b01951c222bd061d6d87c48903836a9", "GPL-2.0-or-later AND LGPL-2.1-or-later", "main"),
    ("libunistring-1.3-r0.apk", "989640e58f7646c0495c3950df36eb7b4df152d8150c2c888f8ab954fed8c908", "GPL-2.0-or-later OR LGPL-3.0-or-later", "main"),
    ("libuv-1.51.0-r0.apk", "6cd879561ba6380c5664c221c0de33af177becb0d94a5de0d5d90589a2089737", "MIT", "main"),
    ("libxml2-2.13.9-r1.apk", "be361698c6f728f492b99550e99e898ae478940fe593d8c11b14c5f3d1b5a938", "MIT", "main"),
    ("linux-headers-6.14.2-r0.apk", "b0d7184f0e8d926961b82dff3d8a6a1f85db100dfc97e3fa1e6a25bbe9fd0f71", "GPL-2.0-only", "main"),
    ("llvm20-libs-20.1.8-r0.apk", "d0926e20a9e2e65f03f673c1d9d2e69f807e5858920146a0eaedaa68b30fe443", "Apache-2.0", "main"),
    ("lz4-libs-1.10.0-r0.apk", "b4f393b9d8ce5a431788b31c899734f84d60a47487a6b833ee8d8012c501fe3f", "BSD-2-Clause AND GPL-2.0-or-later", "main"),
    ("make-4.4.1-r3.apk", "5b66eb4c2db0b0452f1c88d19909397f09e72425d1d9769a480d6370d8a9c310", "GPL-3.0-or-later", "main"),
    ("mpc1-1.3.1-r1.apk", "bfc0d8bd7e3b8de2fdfcf4977437281e58147cc568194cb82ba887bd190376e6", "LGPL-3.0-or-later", "main"),
    ("mpfr4-4.2.1_p1-r0.apk", "33f2f274333164dfc25a0002ca58bb99f6d26843d5486c03e3fa207e5432ac59", "LGPL-3.0-or-later", "main"),
    ("musl-1.2.5-r12.apk", "4990a5e0ba312e478f94cfe431a70efef1538004eb361c8ae424516848be45bb", "MIT", "main"),
    ("musl-dev-1.2.5-r12.apk", "1c2068d910cfdbbcb4eb107a5a478f8b48edf3a6311953c8fa5180c5190efab3", "MIT", "main"),
    ("nghttp2-libs-1.69.0-r0.apk", "d6ee515d30d703e94b55ef9c0a02aea053313f654acbd2c655d18e3907ecb66a", "MIT", "main"),
    ("openblas-0.3.28-r0.apk", "d84c38981a06ffc9070618d130e6bd5548f7ad109aed22f4769fa9b55beae036", "BSD-3-Clause", "community"),
    ("openblas-dev-0.3.28-r0.apk", "9c02593e0bf9ca6480bfb4e82e87badef8e0701daeb01b19bbe9c16ab105a9d4", "BSD-3-Clause", "community"),
    ("openblas-ilp64-0.3.28-r0.apk", "c978ff60522ae5a12f3c171a7be20f0bf88eade0842fbbc5b67456e1cd04a0bd", "BSD-3-Clause", "community"),
    ("patch-2.8-r0.apk", "f34098dde672a06d7b601c12d2e585bbf473292dd08d7e0a81e9ad27962196a6", "GPL-3.0-or-later", "main"),
    ("pkgconf-2.4.3-r0.apk", "32ed811fa76f5f380cda11aa5bdf2f93727760dc4c6d6de5f06dd4c66e3fe3ba", "ISC", "main"),
    ("rhash-libs-1.4.5-r0.apk", "c48bd5cf228e606e961f07a67bf08cfae60136cf93f86f5005678ea7b374bfb9", "0BSD", "main"),
    ("rust-1.87.0-r1.apk", "6ec7b081309981b4d1188781e5e37ce5c2215697f8755fbac8bcc717987a0608", "Apache-2.0 OR MIT", "main"),
    ("scudo-malloc-20.1.8-r0.apk", "e83a392d02dde86fe09da81a97ed29a279f8af1e0ba0f7cf2cbcae88a57c3f6f", "Apache-2.0 WITH LLVM-exception", "main"),
    ("xz-libs-5.8.3-r0.apk", "ebb49f0e8efb6ac774d5d8f21a7cf7d63c4658fa36ab1ebaf48ebd67efd60142", "GPL-2.0-or-later AND 0BSD AND Public-Domain AND LGPL-2.1-or-later", "main"),
    ("zlib-1.3.2-r0.apk", "1f3d5f463f490dad3a68097376711bfe5e8156e9e8daff3070513aa4378cdeca", "Zlib", "main"),
    ("zstd-libs-1.5.7-r0.apk", "1bdd6e57cfbfbfd6e8481cad37ddd5d199950715bec1879b3afb600272dbb09e", "BSD-3-Clause OR GPL-2.0-or-later", "main"),
]

# Standalone tools
PINNED_BUN_URL = "https://github.com/oven-sh/bun/releases/download/bun-v1.4.0/bun-linux-x64-musl.zip"
PINNED_BUN_SHA256 = "83b5f12fd258dd8d4fdcaea65ede954366aa717dab399e20093ecab280d54e7a"
PINNED_BUN_VERSION = "bun-v1.4.0"

PINNED_UV_URL = "https://github.com/astral-sh/uv/releases/download/0.12.9/uv-x86_64-unknown-linux-musl.tar.gz"
PINNED_UV_SHA256 = "aa4b1f8770910f7c7c543c7acc980e4270e52e70750c996acef813ea1c7c2912"
PINNED_UV_VERSION = "0.12.9"

PINNED_PYTHON_URL = "https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.11.16%2B20260901-x86_64-unknown-linux-musl-install_only.tar.gz"
PINNED_PYTHON_SHA256 = "dd9d55e2d83a90097b6c5c9d3b770e064f91278cf176a8f8369c11dcf1a4bc10"
PINNED_PYTHON_VERSION = "cpython-3.11.16+20260901"

PINNED_NINJA_URL = "https://files.pythonhosted.org/packages/f5/5f/c511f2952f94ab2966d60edd9c34e744ea32f2724b1184b62270bde55b3a/ninja-1.13.2-py3-none-musllinux_1_2_x86_64.whl"
PINNED_NINJA_SHA256 = "915bd482c4be41c75120fd67a22e0bb3f0fbb3bbc5f95b89787deadd59e27ef2"
PINNED_NINJA_VERSION = "1.13.2"

STANDALONE_TOOLS: dict[str, dict[str, Any]] = {
    "bun": {
        "version": PINNED_BUN_VERSION,
        "source_url": PINNED_BUN_URL,
        "sha256": PINNED_BUN_SHA256,
        "license": "MIT",
        "archive_format": "zip",
        "target_bin": "bun",
        "extract_path": "bun-linux-x64-musl/bun",
        "feasibility": "feasible",
        "feasibility_reason": "Official Bun Linux musl binary documented for Alpine Linux x86_64.",
    },
    "uv": {
        "version": PINNED_UV_VERSION,
        "source_url": PINNED_UV_URL,
        "sha256": PINNED_UV_SHA256,
        "license": "Apache-2.0 OR MIT",
        "archive_format": "tar.gz",
        "target_bin": "uv",
        "extract_path": "uv-x86_64-unknown-linux-musl/uv",
        "feasibility": "feasible",
        "feasibility_reason": "Official Astral uv static/musl release binary for x86_64-unknown-linux-musl.",
    },
    "ninja": {
        "version": PINNED_NINJA_VERSION,
        "source_url": PINNED_NINJA_URL,
        "sha256": PINNED_NINJA_SHA256,
        "license": "Apache-2.0",
        "archive_format": "zip",
        "target_bin": "ninja",
        "extract_path": "ninja-1.13.2.data/scripts/ninja",
        "feasibility": "feasible",
        "feasibility_reason": "Official Ninja musl binary wheel for x86_64.",
    },
    "python3": {
        "version": PINNED_PYTHON_VERSION,
        "source_url": PINNED_PYTHON_URL,
        "sha256": PINNED_PYTHON_SHA256,
        "license": "PSF-2.0",
        "archive_format": "tar.gz",
        "target_bin": "python3",
        "extract_path": "python",  # Full runtime tree
        "feasibility": "feasible",
        "feasibility_reason": "Astral python-build-standalone official musl install-only build for Linux x86_64.",
    },
}

ALPINE_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "curl": {
        "package": "curl-8.14.1-r3.apk",
        "version": "8.14.1-r3",
        "license": "curl",
        "target_bin": "curl",
        "rel_path": "usr/bin/curl",
        "needs_alpine_libs": True,
        "needs_ca_bundle": True,
    },
    "git": {
        "package": "git-2.49.1-r0.apk",
        "version": "2.49.1-r0",
        "license": "GPL-2.0-only",
        "target_bin": "git",
        "rel_path": "usr/bin/git",
        "needs_alpine_libs": True,
        "needs_git_paths": True,
        "needs_ca_bundle": True,
    },
    "jq": {
        "package": "jq-1.8.2-r0.apk",
        "version": "1.8.2-r0",
        "license": "MIT",
        "target_bin": "jq",
        "rel_path": "usr/bin/jq",
        "needs_alpine_libs": True,
    },
    "sqlite3": {
        "package": "sqlite-3.49.2-r1.apk",
        "version": "3.49.2-r1",
        "license": "blessing",
        "target_bin": "sqlite3",
        "rel_path": "usr/bin/sqlite3",
        "needs_alpine_libs": True,
    },
    "rsync": {
        "package": "rsync-3.5.0-r0.apk",
        "version": "3.5.0-r0",
        "license": "GPL-3.0-or-later",
        "target_bin": "rsync",
        "rel_path": "usr/bin/rsync",
        "needs_alpine_libs": True,
    },
    "zstd": {
        "package": "zstd-1.5.7-r0.apk",
        "version": "1.5.7-r0",
        "license": "BSD-3-Clause OR GPL-2.0-or-later",
        "target_bin": "zstd",
        "rel_path": "usr/bin/zstd",
        "needs_alpine_libs": True,
    },
}


def redact_url(url: str) -> str:
    """Redact passwords, usernames, and query parameter secrets from URL."""
    try:
        parsed = urllib.parse.urlsplit(url)
    except Exception:
        return re.sub(r":([^/@]+)@", ":[REDACTED]@", url)

    netloc = parsed.netloc
    if "@" in netloc:
        userinfo, host = netloc.split("@", 1)
        if ":" in userinfo:
            user, _ = userinfo.split(":", 1)
            userinfo = f"{user}:[REDACTED]"
        else:
            userinfo = "[REDACTED]"
        netloc = f"{userinfo}@{host}"

    query = parsed.query
    if query:
        pairs = urllib.parse.parse_qsl(query, keep_blank_values=True)
        redacted_pairs = []
        for k, v in pairs:
            k_lower = k.lower()
            if any(s in k_lower for s in ("token", "key", "secret", "pass", "auth", "sig")):
                redacted_pairs.append((k, "[REDACTED]"))
            else:
                redacted_pairs.append((k, v))
        query = urllib.parse.urlencode(redacted_pairs)

    return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))


def sanitize_text(text: str) -> str:
    """Sanitize arbitrary error/exception text to redact credentials and secret tokens."""
    # Redact user:password@
    redacted = re.sub(r":([^/@:\s'\"`]+)@([^\s/'\"`]+)", r":[REDACTED]@\2", text)
    # Redact nonnumeric port: 'password@host' or 'password' when urllib treats user:pass@host incorrectly
    redacted = re.sub(r"(['\"]?)(?:[^/@:\s'\"`]+:)?([^/@:\s'\"`]+)@([^\s/'\"`]+)(['\"]?)", r"\1[REDACTED]@\3\4", redacted)
    # Redact query parameters with sensitive names
    redacted = re.sub(
        r"([?&](?:token|key|secret|pass|password|auth|sig)[^=]*=)([^& \t\n\r'\"`]+)",
        r"\1[REDACTED]",
        redacted,
        flags=re.IGNORECASE,
    )
    return redacted


def is_acceptable_manifest_url(url: str) -> bool:
    """Manifest-recorded URLs are accepted only as well-formed HTTPS URLs with no visible credentials or query secrets."""
    try:
        parsed = urllib.parse.urlsplit(url)
    except Exception:
        return False
    if parsed.scheme != "https" or not parsed.netloc:
        return False
    if parsed.username or parsed.password:
        return False
    if parsed.query:
        for k, _v in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
            if any(s in k.lower() for s in ("token", "key", "secret", "pass", "auth", "sig")):
                return False
    return True


def die(msg: str) -> None:
    sys.stderr.write(f"ERROR: {msg}\n")
    sys.exit(1)


def check_forbidden_commands(args_list: list[str]) -> None:
    """Reject actual commands and injections for apk, sudo, docker without blocking .apk filenames."""
    env_inject = os.environ.get("PORTFOLIO_LAB_BOOTSTRAP_INJECT", "")
    items_to_check = [*args_list, env_inject]

    # Command patterns that represent actual tool execution intent
    forbidden_patterns = [
        re.compile(r"\bapk\s+(add|del|update|upgrade|fix)\b", re.IGNORECASE),
        re.compile(r"(^|[\s;&|`$])apk\b(?![-._a-zA-Z0-9])", re.IGNORECASE),
        re.compile(r"\bsudo\b", re.IGNORECASE),
        re.compile(r"\bdocker\b", re.IGNORECASE),
    ]

    for item in items_to_check:
        if not item:
            continue
        # Allow legitimate .apk package arguments or file paths
        if item.endswith(".apk") or ".apk=" in item or "/alpine/v3.22/" in item:
            continue
        for pat in forbidden_patterns:
            if pat.search(item):
                die(f"forbidden command or token detected in '{item}'. Root/apk/sudo/docker operations are strictly prohibited.")


def check_root_execution(prefix_str: str) -> None:
    force_root = os.environ.get("PORTFOLIO_LAB_BOOTSTRAP_FORCE_ROOT", "") == "1"
    is_root = (hasattr(os, "geteuid") and os.geteuid() == 0) or force_root
    if is_root:
        if prefix_str == DEFAULT_PREFIX or prefix_str in ("/usr", "/usr/local", "/etc", "/var", "/bin", "/sbin"):
            die("refusing to run as root: root execution would mutate base-container paths.")


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class BootstrapManager:
    def __init__(
        self,
        prefix: Path,
        *,
        url_overrides: dict[str, str] | None = None,
        artifact_overrides: dict[str, Path] | None = None,
        sha_overrides: dict[str, str] | None = None,
        skip_tools: set[str] | None = None,
        mock_closure: bool = False,
        network_verify: bool = False,
    ) -> None:
        self.prefix = prefix
        self.bin_dir = prefix / "bin"
        self.toolchain_dir = prefix / DEFAULT_TOOLCHAIN_SUBDIR
        self.alpine_root = prefix / DEFAULT_ALPINE_ROOT_SUBDIR
        self.alpine_build_root = prefix / DEFAULT_ALPINE_BUILD_ROOT_SUBDIR
        self.python_root = prefix / DEFAULT_PYTHON_ROOT_SUBDIR
        self.standalone_dir = prefix / DEFAULT_STANDALONE_SUBDIR
        self.manifest_path = self.toolchain_dir / "bootstrap-manifest.json"
        self.url_overrides = url_overrides or {}
        self.artifact_overrides = artifact_overrides or {}
        self.sha_overrides = sha_overrides or {}
        self.skip_tools = skip_tools or set()
        self.mock_closure = mock_closure
        self.network_verify = network_verify

        self._validate_prefix()
        self._validate_overrides()

    def _validate_prefix(self) -> None:
        raw_str = str(self.prefix)
        parts = Path(raw_str).parts
        if ".." in parts:
            die(f"prefix contains traversal path: {self.prefix}")

    def _validate_overrides(self) -> None:
        known = set(STANDALONE_TOOLS.keys()) | set(ALPINE_TOOL_SPECS.keys()) | {"bunx"}
        known |= {pkg for pkg, _, _ in ALPINE_PACKAGE_CLOSURE}
        known |= {pkg for pkg, _, _, _ in ALPINE_BUILD_PACKAGE_CLOSURE}
        for k in list(self.url_overrides.keys()) + list(self.artifact_overrides.keys()) + list(self.sha_overrides.keys()):
            if k not in known:
                die(f"unknown override tool/package name: {k}")

    def build_manifest(self) -> dict[str, Any]:
        packages_dict: dict[str, Any] = {}
        pkg_closure = [
            (pkg, sha, lic)
            for (pkg, sha, lic) in ALPINE_PACKAGE_CLOSURE
            if (not self.mock_closure) or (pkg in self.artifact_overrides)
        ]

        for pkg_name, default_sha, lic in pkg_closure:
            default_url = f"{ALPINE_DL_BASE}/{pkg_name}"
            url = self.url_overrides.get(pkg_name, default_url)
            sha = self.sha_overrides.get(pkg_name, default_sha)
            packages_dict[pkg_name] = {
                "source_url": redact_url(url),
                "sha256": sha,
                "license": lic,
                "install_root": str(self.alpine_root),
                "repository": "main",
            }

        build_packages_dict: dict[str, Any] = {}
        build_pkg_closure = [
            (pkg, sha, lic, repo)
            for (pkg, sha, lic, repo) in ALPINE_BUILD_PACKAGE_CLOSURE
            if (not self.mock_closure) or (pkg in self.artifact_overrides)
        ]

        for pkg_name, default_sha, lic, repo in build_pkg_closure:
            base_url = ALPINE_COMMUNITY_DL_BASE if repo == "community" else ALPINE_DL_BASE
            default_url = f"{base_url}/{pkg_name}"
            url = self.url_overrides.get(pkg_name, default_url)
            sha = self.sha_overrides.get(pkg_name, default_sha)
            build_packages_dict[pkg_name] = {
                "source_url": redact_url(url),
                "sha256": sha,
                "license": lic,
                "install_root": str(self.alpine_build_root),
                "repository": repo,
            }

        tools_dict: dict[str, Any] = {}
        for name, spec in sorted(STANDALONE_TOOLS.items()):
            if name in self.skip_tools:
                continue
            url = self.url_overrides.get(name, spec["source_url"])
            sha = self.sha_overrides.get(name, spec["sha256"])
            tools_dict[name] = {
                "version": spec["version"],
                "source_url": redact_url(url),
                "sha256": sha,
                "license": spec["license"],
                "archive_format": spec["archive_format"],
                "install_path": str(self.bin_dir / spec["target_bin"]),
                "uninstall_path": str(self.bin_dir / spec["target_bin"]),
                "feasibility": spec["feasibility"],
                "feasibility_reason": spec["feasibility_reason"],
            }

        if "bun" not in self.skip_tools and "bunx" not in self.skip_tools:
            tools_dict["bunx"] = {
                "version": PINNED_BUN_VERSION,
                "source_url": redact_url(self.url_overrides.get("bun", PINNED_BUN_URL)),
                "sha256": self.sha_overrides.get("bun", PINNED_BUN_SHA256),
                "license": "MIT",
                "archive_format": "wrapper",
                "install_path": str(self.bin_dir / "bunx"),
                "uninstall_path": str(self.bin_dir / "bunx"),
                "feasibility": "feasible",
                "feasibility_reason": "Relocatable managed wrapper delegating to bun x.",
            }

        for name, spec in sorted(ALPINE_TOOL_SPECS.items()):
            if name in self.skip_tools:
                continue
            pkg_name = spec["package"]
            default_url = f"{ALPINE_DL_BASE}/{pkg_name}"
            url = self.url_overrides.get(name, self.url_overrides.get(pkg_name, default_url))
            sha = self.sha_overrides.get(name, packages_dict.get(pkg_name, {}).get("sha256", ""))
            tools_dict[name] = {
                "version": spec["version"],
                "source_url": redact_url(url),
                "sha256": sha,
                "license": spec["license"],
                "archive_format": "apk",
                "install_path": str(self.bin_dir / spec["target_bin"]),
                "uninstall_path": str(self.bin_dir / spec["target_bin"]),
                "feasibility": "feasible",
                "feasibility_reason": f"Installed natively into user alpine-root from Alpine v3.22 {pkg_name} closure.",
            }

        return {
            "schema_version": SCHEMA_VERSION,
            "layout_revision": LAYOUT_REVISION,
            "prefix": str(self.prefix),
            "bin_dir": str(self.bin_dir),
            "toolchain_dir": str(self.toolchain_dir),
            "alpine_root": str(self.alpine_root),
            "alpine_build_root": str(self.alpine_build_root),
            "python_root": str(self.python_root),
            "standalone_dir": str(self.standalone_dir),
            "tools": tools_dict,
            "packages": packages_dict,
            "build_packages": build_packages_dict,
        }

    def dry_run(self) -> None:
        manifest = self.build_manifest()
        print(json.dumps(manifest, indent=2, sort_keys=True))

    def _safe_extract_apk(self, apk_path: Path, target_dir: Path) -> None:
        """Extract APK (tar.gz) checking for path traversal, absolute paths, hardlinks, and escaping symlinks."""
        resolved_target = target_dir.resolve()
        with tarfile.open(apk_path, "r:gz") as tf:
            for member in tf.getmembers():
                name = member.name
                if name.startswith("/") or name.startswith("\\"):
                    die(f"unsafe absolute path in archive {apk_path.name}: {name}")
                if ".." in Path(name).parts:
                    die(f"unsafe traversal path in archive {apk_path.name}: {name}")

                dest_path = (target_dir / name).resolve()
                if not (dest_path == resolved_target or dest_path.is_relative_to(resolved_target)):
                    die(f"archive member escapes target destination: {name}")

                if member.isreg():
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    if dest_path.is_symlink() or dest_path.exists():
                        dest_path.unlink()
                    f = tf.extractfile(member)
                    if f is not None:
                        dest_path.write_bytes(f.read())
                        dest_path.chmod(member.mode | stat.S_IRUSR | stat.S_IWUSR)
                elif member.isdir():
                    dest_path.mkdir(parents=True, exist_ok=True)
                elif member.islnk():
                    # Hardlink: translate and ensure target exists inside target_dir
                    link_source = (target_dir / member.linkname).resolve()
                    if not (link_source == resolved_target or link_source.is_relative_to(resolved_target)):
                        die(f"unsafe hardlink escape in {apk_path.name}: {name} -> {member.linkname}")
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    if dest_path.exists():
                        dest_path.unlink()
                    os.link(link_source, dest_path)
                elif member.issym():
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    raw_target = member.linkname
                    if raw_target.startswith("/"):
                        # Rewrite absolute package symlinks so they remain contained under target_dir
                        rel_target = os.path.relpath(target_dir / raw_target.lstrip("/"), dest_path.parent)
                        link_value = rel_target
                    else:
                        link_value = raw_target

                    # Verify final symlink resolution containment
                    check_target = (dest_path.parent / link_value).resolve()
                    if not (check_target == resolved_target or check_target.is_relative_to(resolved_target)):
                        die(f"unsafe symlink escape in {apk_path.name}: {name} -> {member.linkname}")

                    if dest_path.is_symlink() or dest_path.exists():
                        dest_path.unlink()
                    os.symlink(link_value, dest_path)

    def _create_alpine_wrapper(self, bin_name: str, spec: dict[str, Any], wrapper_path: Path) -> None:
        """Generate a relocatable executable wrapper configuring private Alpine LD_LIBRARY_PATH and CA bundle."""
        target_bin_rel = spec["rel_path"]

        extra_env = ""
        if spec.get("needs_git_paths"):
            extra_env += """
export GIT_EXEC_PATH="${PORTFOLIO_LAB_GIT_EXEC_PATH:-$ALPINE_ROOT/usr/libexec/git-core}"
export GIT_TEMPLATE_DIR="${PORTFOLIO_LAB_GIT_TEMPLATE_DIR:-$ALPINE_ROOT/usr/share/git-core/templates}"
"""
        if spec.get("needs_ca_bundle"):
            extra_env += """
CA_BUNDLE="$ALPINE_ROOT/etc/ssl/certs/ca-certificates.crt"
if [ -f "$CA_BUNDLE" ]; then
    export SSL_CERT_FILE="${SSL_CERT_FILE:-$CA_BUNDLE}"
    export CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-$CA_BUNDLE}"
    export GIT_SSL_CAINFO="${GIT_SSL_CAINFO:-$CA_BUNDLE}"
fi
"""

        script = f"""#!/bin/sh
# Portfolio Lab user-owned wrapper for {bin_name} (relocatable)
set -eu

BIN_DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX="$(cd "$BIN_DIR/.." && pwd)"
ALPINE_ROOT="${{PORTFOLIO_LAB_ALPINE_ROOT:-$PREFIX/{DEFAULT_ALPINE_ROOT_SUBDIR}}}"
AL_LIB="$ALPINE_ROOT/lib:$ALPINE_ROOT/usr/lib"

if [ -n "${{LD_LIBRARY_PATH:-}}" ]; then
    export LD_LIBRARY_PATH="$AL_LIB:$LD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$AL_LIB"
fi
{extra_env}
exec "$ALPINE_ROOT/{target_bin_rel}" "$@"
"""
        wrapper_path.write_text(script, encoding="utf-8")
        wrapper_path.chmod(0o755)

    def _create_python_wrapper(self, wrapper_path: Path) -> None:
        """Generate a relocatable wrapper for standalone Python with native OpenBLAS / build-root library support."""
        script = f"""#!/bin/sh
# Portfolio Lab user-owned wrapper for python3 (relocatable)
set -eu

BIN_DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX="$(cd "$BIN_DIR/.." && pwd)"
PYTHON_ROOT="${{PORTFOLIO_LAB_PYTHON_ROOT:-$PREFIX/{DEFAULT_PYTHON_ROOT_SUBDIR}}}"
ALPINE_ROOT="${{PORTFOLIO_LAB_ALPINE_ROOT:-$PREFIX/{DEFAULT_ALPINE_ROOT_SUBDIR}}}"
BUILD_ROOT="${{PORTFOLIO_LAB_ALPINE_BUILD_ROOT:-$PREFIX/{DEFAULT_ALPINE_BUILD_ROOT_SUBDIR}}}"

LIB_DIRS="$BUILD_ROOT/usr/lib:$BUILD_ROOT/lib:$ALPINE_ROOT/usr/lib:$ALPINE_ROOT/lib"
if [ -n "${{LD_LIBRARY_PATH:-}}" ]; then
    export LD_LIBRARY_PATH="$LIB_DIRS:$LD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$LIB_DIRS"
fi

exec "$PYTHON_ROOT/bin/python3" "$@"
"""
        wrapper_path.write_text(script, encoding="utf-8")
        wrapper_path.chmod(0o755)

    def _create_bun_wrapper(self, wrapper_path: Path) -> None:
        """Generate a relocatable wrapper for Bun configuring Alpine LD_LIBRARY_PATH and private CA variables."""
        script = f"""#!/bin/sh
# Portfolio Lab user-owned wrapper for Bun (relocatable)
set -eu

BIN_DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX="$(cd "$BIN_DIR/.." && pwd)"
ALPINE_ROOT="${{PORTFOLIO_LAB_ALPINE_ROOT:-$PREFIX/{DEFAULT_ALPINE_ROOT_SUBDIR}}}"
AL_LIB="$ALPINE_ROOT/lib:$ALPINE_ROOT/usr/lib"

if [ -n "${{LD_LIBRARY_PATH:-}}" ]; then
    export LD_LIBRARY_PATH="$AL_LIB:$LD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$AL_LIB"
fi

CA_BUNDLE="$ALPINE_ROOT/etc/ssl/certs/ca-certificates.crt"
if [ -f "$CA_BUNDLE" ]; then
    export SSL_CERT_FILE="${{SSL_CERT_FILE:-$CA_BUNDLE}}"
    export NODE_EXTRA_CA_CERTS="${{NODE_EXTRA_CA_CERTS:-$CA_BUNDLE}}"
fi

STANDALONE_BUN="$PREFIX/{DEFAULT_STANDALONE_SUBDIR}/bun"
exec "$STANDALONE_BUN" "$@"
"""
        wrapper_path.write_text(script, encoding="utf-8")
        wrapper_path.chmod(0o755)

    def _create_bunx_wrapper(self, wrapper_path: Path) -> None:
        """Generate a relocatable wrapper for bunx delegating to bun x."""
        script = """#!/bin/sh
# Portfolio Lab user-owned wrapper for bunx (relocatable)
set -eu

BIN_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$BIN_DIR/bun" x "$@"
"""
        wrapper_path.write_text(script, encoding="utf-8")
        wrapper_path.chmod(0o755)

    def _create_ninja_wrapper(self, wrapper_path: Path) -> None:
        """Generate a relocatable wrapper for Ninja delegating to standalone binary."""
        script = f"""#!/bin/sh
# Portfolio Lab user-owned wrapper for Ninja (relocatable)
set -eu

BIN_DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX="$(cd "$BIN_DIR/.." && pwd)"
STANDALONE_NINJA="$PREFIX/{DEFAULT_STANDALONE_SUBDIR}/ninja"
exec "$STANDALONE_NINJA" "$@"
"""
        wrapper_path.write_text(script, encoding="utf-8")
        wrapper_path.chmod(0o755)

    def get_build_environment(self) -> dict[str, str]:
        """Construct the proven relocatable native build environment matching the uv wrapper."""
        build_root = self.alpine_build_root
        alpine_root = self.alpine_root
        bin_dir = self.bin_dir
        standalone_dir = self.standalone_dir

        build_libs = f"{build_root}/usr/lib:{build_root}/lib"
        al_libs = f"{alpine_root}/usr/lib:{alpine_root}/lib"
        all_libs = f"{build_libs}:{al_libs}"

        env = dict(os.environ)

        # Path ordering: bin_dir, then standalone_dir (Ninja), then build-root bins, then existing PATH
        env["PATH"] = f"{bin_dir}:{standalone_dir}:{build_root}/usr/bin:{build_root}/bin:{env.get('PATH', '')}"

        # Library search paths
        if "LD_LIBRARY_PATH" in env and env["LD_LIBRARY_PATH"]:
            env["LD_LIBRARY_PATH"] = f"{all_libs}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = all_libs

        if "LIBRARY_PATH" in env and env["LIBRARY_PATH"]:
            env["LIBRARY_PATH"] = f"{all_libs}:{env['LIBRARY_PATH']}"
        else:
            env["LIBRARY_PATH"] = all_libs

        # Compilers and binutils: managed paths only.
        # Generic ambient CC/CXX/FC/AR/RANLIB must not substitute host tools.
        # Operator overrides allowed only via explicit PORTFOLIO_LAB_* variables.
        env["CC"] = os.environ.get("PORTFOLIO_LAB_CC", f"{build_root}/usr/bin/gcc")
        env["CXX"] = os.environ.get("PORTFOLIO_LAB_CXX", f"{build_root}/usr/bin/g++")
        env["FC"] = os.environ.get("PORTFOLIO_LAB_FC", f"{build_root}/usr/bin/gfortran")
        env["AR"] = os.environ.get("PORTFOLIO_LAB_AR", f"{build_root}/usr/bin/ar")
        env["RANLIB"] = os.environ.get("PORTFOLIO_LAB_RANLIB", f"{build_root}/usr/bin/ranlib")

        # Headers and pkg-config/cmake paths
        env["C_INCLUDE_PATH"] = f"{build_root}/usr/include"
        env["CPLUS_INCLUDE_PATH"] = (
            f"{build_root}/usr/include/c++/14.2.0:"
            f"{build_root}/usr/include/c++/14.2.0/x86_64-alpine-linux-musl:"
            f"{build_root}/usr/include"
        )
        env["PKG_CONFIG_PATH"] = f"{build_root}/usr/lib/pkgconfig:{build_root}/usr/share/pkgconfig"
        env["CMAKE_PREFIX_PATH"] = f"{build_root}/usr"

        # Compiler and linker flags
        cflags_base = f"-I{build_root}/usr/include"
        ldflags_base = f"-L{build_root}/usr/lib -L{build_root}/lib -Wl,-rpath-link,{build_root}/usr/lib:{build_root}/lib"
        env["CFLAGS"] = f"{env.get('CFLAGS', '')} {cflags_base}".strip()
        env["CXXFLAGS"] = f"{env.get('CXXFLAGS', '')} {cflags_base}".strip()
        env["LDFLAGS"] = f"{env.get('LDFLAGS', '')} {ldflags_base}".strip()

        # Private CA bundle
        ca_bundle = alpine_root / "etc" / "ssl" / "certs" / "ca-certificates.crt"
        if ca_bundle.is_file():
            env.setdefault("SSL_CERT_FILE", str(ca_bundle))
            env.setdefault("CURL_CA_BUNDLE", str(ca_bundle))
            env.setdefault("GIT_SSL_CAINFO", str(ca_bundle))

        return env

    def _create_uv_wrapper(self, wrapper_path: Path) -> None:
        """Generate a relocatable wrapper for uv configuring the proven Alpine native build environment."""
        script = f"""#!/bin/sh
# Portfolio Lab user-owned wrapper for uv (relocatable)
set -eu

BIN_DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX="$(cd "$BIN_DIR/.." && pwd)"
ALPINE_ROOT="${{PORTFOLIO_LAB_ALPINE_ROOT:-$PREFIX/{DEFAULT_ALPINE_ROOT_SUBDIR}}}"
BUILD_ROOT="${{PORTFOLIO_LAB_ALPINE_BUILD_ROOT:-$PREFIX/{DEFAULT_ALPINE_BUILD_ROOT_SUBDIR}}}"
STANDALONE_DIR="$PREFIX/{DEFAULT_STANDALONE_SUBDIR}"

# Prepend BIN_DIR before standalone/build-root paths, then standalone Ninja, then build-root bins, then host PATH
export PATH="$BIN_DIR:$STANDALONE_DIR:$BUILD_ROOT/usr/bin:$BUILD_ROOT/bin:$PATH"

# Library search paths
BUILD_LIBS="$BUILD_ROOT/usr/lib:$BUILD_ROOT/lib"
AL_LIBS="$ALPINE_ROOT/usr/lib:$ALPINE_ROOT/lib"
ALL_LIBS="$BUILD_LIBS:$AL_LIBS"

if [ -n "${{LD_LIBRARY_PATH:-}}" ]; then
    export LD_LIBRARY_PATH="$ALL_LIBS:$LD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$ALL_LIBS"
fi

if [ -n "${{LIBRARY_PATH:-}}" ]; then
    export LIBRARY_PATH="$ALL_LIBS:$LIBRARY_PATH"
else
    export LIBRARY_PATH="$ALL_LIBS"
fi

# Compiler and header search paths for Alpine GCC 14.2
# Generic ambient CC/CXX/FC/AR/RANLIB are intentionally ignored; managed paths are default.
export CC="${{PORTFOLIO_LAB_CC:-$BUILD_ROOT/usr/bin/gcc}}"
export CXX="${{PORTFOLIO_LAB_CXX:-$BUILD_ROOT/usr/bin/g++}}"
export FC="${{PORTFOLIO_LAB_FC:-$BUILD_ROOT/usr/bin/gfortran}}"
export AR="${{PORTFOLIO_LAB_AR:-$BUILD_ROOT/usr/bin/ar}}"
export RANLIB="${{PORTFOLIO_LAB_RANLIB:-$BUILD_ROOT/usr/bin/ranlib}}"

export C_INCLUDE_PATH="$BUILD_ROOT/usr/include"
export CPLUS_INCLUDE_PATH="$BUILD_ROOT/usr/include/c++/14.2.0:$BUILD_ROOT/usr/include/c++/14.2.0/x86_64-alpine-linux-musl:$BUILD_ROOT/usr/include"

export PKG_CONFIG_PATH="$BUILD_ROOT/usr/lib/pkgconfig:$BUILD_ROOT/usr/share/pkgconfig"
export CMAKE_PREFIX_PATH="$BUILD_ROOT/usr"

# Sysroot compiler/linker flags
CFLAGS_BASE="-I$BUILD_ROOT/usr/include"
LDFLAGS_BASE="-L$BUILD_ROOT/usr/lib -L$BUILD_ROOT/lib -Wl,-rpath-link,$BUILD_ROOT/usr/lib:$BUILD_ROOT/lib"

export CFLAGS="${{CFLAGS:-}} $CFLAGS_BASE"
export CXXFLAGS="${{CXXFLAGS:-}} $CFLAGS_BASE"
export LDFLAGS="${{LDFLAGS:-}} $LDFLAGS_BASE"

# Private CA
CA_BUNDLE="$ALPINE_ROOT/etc/ssl/certs/ca-certificates.crt"
if [ -f "$CA_BUNDLE" ]; then
    export SSL_CERT_FILE="${{SSL_CERT_FILE:-$CA_BUNDLE}}"
    export CURL_CA_BUNDLE="${{CURL_CA_BUNDLE:-$CA_BUNDLE}}"
    export GIT_SSL_CAINFO="${{GIT_SSL_CAINFO:-$CA_BUNDLE}}"
fi

STANDALONE_UV="$STANDALONE_DIR/uv"
exec "$STANDALONE_UV" "$@"
"""
        wrapper_path.write_text(script, encoding="utf-8")
        wrapper_path.chmod(0o755)

    def install(self) -> None:
        manifest = self.build_manifest()
        tools = manifest["tools"]

        # Idempotency check: verify schema, layout revision, directories, and tool/package metadata
        if self.manifest_path.is_file():
            try:
                existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                existing_tools = existing.get("tools", {})
                existing_packages = existing.get("packages", {})
                existing_layout = existing.get("layout_revision")
                existing_schema = existing.get("schema_version")

                up_to_date = (
                    existing_schema == SCHEMA_VERSION
                    and existing_layout == LAYOUT_REVISION
                    and ("python3" in self.skip_tools or self.python_root.is_dir())
                    and self.alpine_root.is_dir()
                    and self.alpine_build_root.is_dir()
                )

                if up_to_date:
                    # If bun is enabled, standalone payload must exist
                    if "bun" not in self.skip_tools:
                        standalone_bun = self.standalone_dir / "bun"
                        if not standalone_bun.is_file():
                            up_to_date = False

                    # If uv is enabled, standalone payload must exist
                    if "uv" not in self.skip_tools:
                        standalone_uv = self.standalone_dir / "uv"
                        if not standalone_uv.is_file():
                            up_to_date = False

                    # If ninja is enabled, standalone payload must exist
                    if "ninja" not in self.skip_tools:
                        standalone_ninja = self.standalone_dir / "ninja"
                        if not standalone_ninja.is_file():
                            up_to_date = False

                    # Check packages closures match
                    expected_pkgs = manifest.get("packages", {})
                    if set(expected_pkgs.keys()) != set(existing_packages.keys()):
                        up_to_date = False
                    else:
                        for pkg_name, exp_pkg in expected_pkgs.items():
                            ex_pkg = existing_packages.get(pkg_name, {})
                            for field in ("sha256", "license", "repository", "source_url", "install_root"):
                                if ex_pkg.get(field) != exp_pkg.get(field):
                                    up_to_date = False
                                    break
                            if not up_to_date:
                                break

                    expected_build_pkgs = manifest.get("build_packages", {})
                    existing_build_pkgs = existing.get("build_packages", {})
                    if set(expected_build_pkgs.keys()) != set(existing_build_pkgs.keys()):
                        up_to_date = False
                    else:
                        for pkg_name, exp_pkg in expected_build_pkgs.items():
                            ex_pkg = existing_build_pkgs.get(pkg_name, {})
                            for field in ("sha256", "license", "repository", "source_url", "install_root"):
                                if ex_pkg.get(field) != exp_pkg.get(field):
                                    up_to_date = False
                                    break
                            if not up_to_date:
                                break

                if up_to_date:
                    for tool_name, tool_info in tools.items():
                        if tool_name not in existing_tools:
                            up_to_date = False
                            break
                        ex_info = existing_tools[tool_name]
                        if ex_info.get("version") != tool_info.get("version") or ex_info.get("sha256") != tool_info.get("sha256"):
                            up_to_date = False
                            break
                        target = Path(tool_info["install_path"])
                        if not target.is_file():
                            up_to_date = False
                            break
                        installed_sha = ex_info.get("installed_sha256")
                        if installed_sha and compute_sha256(target) != installed_sha:
                            up_to_date = False
                            break
                        # Check payload sha if recorded
                        if "payload_sha256" in ex_info:
                            if tool_name == "bun":
                                if compute_sha256(self.standalone_dir / "bun") != ex_info["payload_sha256"]:
                                    up_to_date = False
                                    break
                            elif tool_name == "uv":
                                if compute_sha256(self.standalone_dir / "uv") != ex_info["payload_sha256"]:
                                    up_to_date = False
                                    break
                            elif tool_name == "ninja":
                                if compute_sha256(self.standalone_dir / "ninja") != ex_info["payload_sha256"]:
                                    up_to_date = False
                                    break
                            elif tool_name == "python3":
                                if compute_sha256(self.python_root / "bin" / "python3") != ex_info["runtime_sha256"]:
                                    up_to_date = False
                                    break

                if up_to_date:
                    print(f"Toolchain already up to date and idempotent at {self.prefix}")
                    return
            except Exception:
                pass

        # Atomic staging directory on the same filesystem
        self.toolchain_dir.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.toolchain_dir))
        try:
            staging_bin = staging_dir / "bin"
            staging_toolchain = staging_dir / "toolchain"
            staging_alpine_root = staging_toolchain / "alpine-root"
            staging_alpine_build_root = staging_toolchain / "alpine-build-root"
            staging_python_root = staging_toolchain / "python-root"
            staging_standalone = staging_toolchain / "standalone"

            staging_bin.mkdir(parents=True, exist_ok=True)
            staging_alpine_root.mkdir(parents=True, exist_ok=True)
            staging_alpine_build_root.mkdir(parents=True, exist_ok=True)
            staging_python_root.mkdir(parents=True, exist_ok=True)
            staging_standalone.mkdir(parents=True, exist_ok=True)

            installed_manifest_tools: dict[str, Any] = {}

            # 1. Install standalone tools
            for name, spec in sorted(STANDALONE_TOOLS.items()):
                if name in self.skip_tools:
                    continue
                tool_info = tools[name]
                expected_sha = tool_info["sha256"]

                if name in self.artifact_overrides:
                    artifact = self.artifact_overrides[name]
                    if not artifact.is_file():
                        die(f"override artifact missing: {artifact}")
                else:
                    raw_url = self.url_overrides.get(name, spec["source_url"])
                    download_path = staging_dir / f"download-{name}"
                    sys.stderr.write(f"Downloading {name} from {redact_url(raw_url)}...\n")
                    try:
                        urllib.request.urlretrieve(raw_url, download_path)
                        artifact = download_path
                    except Exception as exc:
                        die(f"download failed for {name}: {sanitize_text(str(exc))}")

                actual_sha = compute_sha256(artifact)
                if actual_sha != expected_sha:
                    die(f"checksum mismatch for {name}: expected {expected_sha}, got {actual_sha}")

                archive_format = spec["archive_format"]
                if name == "bun":
                    # Bun payload goes into standalone/bun, wrapper in bin/bun
                    standalone_bun = staging_standalone / "bun"
                    extract_path = spec["extract_path"]
                    with zipfile.ZipFile(artifact, "r") as zf:
                        try:
                            data = zf.read(extract_path)
                        except KeyError:
                            matched = [n for n in zf.namelist() if n.endswith("/bun") or n == "bun"]
                            if not matched:
                                die(f"cannot find {extract_path} in zip {artifact}")
                            data = zf.read(matched[0])
                    standalone_bun.write_bytes(data)
                    standalone_bun.chmod(0o755)

                    bun_wrapper = staging_bin / "bun"
                    self._create_bun_wrapper(bun_wrapper)
                    installed_manifest_tools[name] = dict(tool_info)
                    installed_manifest_tools[name]["installed_sha256"] = compute_sha256(bun_wrapper)
                    installed_manifest_tools[name]["payload_sha256"] = compute_sha256(standalone_bun)

                    if "bunx" in tools:
                        bunx_wrapper = staging_bin / "bunx"
                        self._create_bunx_wrapper(bunx_wrapper)
                        installed_manifest_tools["bunx"] = dict(tools["bunx"])
                        installed_manifest_tools["bunx"]["installed_sha256"] = compute_sha256(bunx_wrapper)
                elif name == "uv":
                    # Standalone uv payload in standalone/uv, wrapper in bin/uv
                    standalone_uv = staging_standalone / "uv"
                    extract_path = spec["extract_path"]
                    with tarfile.open(artifact, "r:gz") as tf:
                        member = None
                        for m in tf.getmembers():
                            if m.name == extract_path or m.name.endswith("/uv"):
                                member = m
                                break
                        if not member:
                            die(f"cannot find {extract_path} in tar {artifact}")
                        f = tf.extractfile(member)
                        if f is None:
                            die(f"failed to extract {extract_path} from {artifact}")
                        standalone_uv.write_bytes(f.read())
                        standalone_uv.chmod(0o755)

                    uv_wrapper = staging_bin / "uv"
                    self._create_uv_wrapper(uv_wrapper)
                    installed_manifest_tools[name] = dict(tool_info)
                    installed_manifest_tools[name]["installed_sha256"] = compute_sha256(uv_wrapper)
                    installed_manifest_tools[name]["payload_sha256"] = compute_sha256(standalone_uv)
                elif name == "ninja":
                    # Standalone Ninja payload in standalone/ninja, wrapper in bin/ninja
                    standalone_ninja = staging_standalone / "ninja"
                    extract_path = spec["extract_path"]
                    with zipfile.ZipFile(artifact, "r") as zf:
                        try:
                            data = zf.read(extract_path)
                        except KeyError:
                            matched = [n for n in zf.namelist() if n.endswith("/ninja") or n == "ninja"]
                            if not matched:
                                die(f"cannot find {extract_path} in zip {artifact}")
                            data = zf.read(matched[0])
                    standalone_ninja.write_bytes(data)
                    standalone_ninja.chmod(0o755)

                    ninja_wrapper = staging_bin / "ninja"
                    self._create_ninja_wrapper(ninja_wrapper)

                    installed_manifest_tools[name] = dict(tool_info)
                    installed_manifest_tools[name]["installed_sha256"] = compute_sha256(ninja_wrapper)
                    installed_manifest_tools[name]["payload_sha256"] = compute_sha256(standalone_ninja)
                elif name == "python3":
                    # Full runtime tree extraction under staging_python_root with safety checks
                    resolved_target = staging_python_root.resolve()
                    with tarfile.open(artifact, "r:gz") as tf:
                        for member in tf.getmembers():
                            m_name = member.name
                            if m_name.startswith("/") or m_name.startswith("\\"):
                                die(f"unsafe absolute path in archive {artifact.name}: {m_name}")
                            parts = Path(m_name).parts
                            if not parts or parts[0] != "python":
                                continue
                            if ".." in parts:
                                die(f"unsafe traversal path in archive {artifact.name}: {m_name}")
                            rel_parts = parts[1:]
                            if not rel_parts:
                                continue
                            rel_name = Path(*rel_parts).as_posix()
                            dest = (staging_python_root / rel_name).resolve()
                            if not (dest == resolved_target or dest.is_relative_to(resolved_target)):
                                die(f"archive member escapes target destination: {m_name}")

                            if member.isdir():
                                dest.mkdir(parents=True, exist_ok=True)
                            elif member.isreg():
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                if dest.is_symlink() or dest.exists():
                                    dest.unlink()
                                f = tf.extractfile(member)
                                if f is not None:
                                    dest.write_bytes(f.read())
                                    dest.chmod(member.mode | stat.S_IRUSR | stat.S_IWUSR)
                            elif member.islnk():
                                # Hardlink: translate and ensure target exists inside staging_python_root
                                link_source = (staging_python_root / member.linkname).resolve()
                                if not (link_source == resolved_target or link_source.is_relative_to(resolved_target)):
                                    die(f"unsafe hardlink escape in {artifact.name}: {m_name} -> {member.linkname}")
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                if dest.exists():
                                    dest.unlink()
                                os.link(link_source, dest)
                            elif member.issym():
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                raw_target = member.linkname
                                if raw_target.startswith("/"):
                                    rel_target = os.path.relpath(staging_python_root / raw_target.lstrip("/"), dest.parent)
                                    link_value = rel_target
                                else:
                                    link_value = raw_target
                                check_target = (dest.parent / link_value).resolve()
                                if not (check_target == resolved_target or check_target.is_relative_to(resolved_target)):
                                    die(f"unsafe symlink escape in {artifact.name}: {m_name} -> {member.linkname}")
                                if dest.is_symlink() or dest.exists():
                                    dest.unlink()
                                os.symlink(link_value, dest)

                    py_wrapper = staging_bin / "python3"
                    self._create_python_wrapper(py_wrapper)
                    installed_manifest_tools[name] = dict(tool_info)
                    installed_manifest_tools[name]["installed_sha256"] = compute_sha256(py_wrapper)
                    installed_manifest_tools[name]["runtime_sha256"] = compute_sha256(staging_python_root / "bin" / "python3")
                else:
                    target_file = staging_bin / spec["target_bin"]
                    extract_path = spec["extract_path"]
                    if archive_format == "zip":
                        with zipfile.ZipFile(artifact, "r") as zf:
                            try:
                                data = zf.read(extract_path)
                            except KeyError:
                                matched = [n for n in zf.namelist() if n.endswith(f"/{spec['target_bin']}") or n == spec["target_bin"]]
                                if not matched:
                                    die(f"cannot find {extract_path} in zip {artifact}")
                                data = zf.read(matched[0])
                        target_file.write_bytes(data)
                        target_file.chmod(0o755)
                    elif archive_format == "tar.gz":
                        with tarfile.open(artifact, "r:gz") as tf:
                            member = None
                            for m in tf.getmembers():
                                if m.name == extract_path or m.name.endswith(f"/{spec['target_bin']}"):
                                    member = m
                                    break
                            if not member:
                                die(f"cannot find {extract_path} in tar {artifact}")
                            f = tf.extractfile(member)
                            if f is None:
                                die(f"failed to extract {extract_path} from {artifact}")
                            target_file.write_bytes(f.read())
                            target_file.chmod(0o755)

                    installed_manifest_tools[name] = dict(tool_info)
                    installed_manifest_tools[name]["installed_sha256"] = compute_sha256(target_file)

            # Preserve managed standalone payloads/wrappers for tools skipped by this partial
            # reinstall: skipping a tool must never destructively remove an already-installed,
            # manifest-verified tool.  Preserved records are carried into the new manifest so
            # verify stays coherent with the surviving files.
            existing_tool_records: dict[str, dict[str, Any]] = {}
            if self.manifest_path.is_file():
                try:
                    existing_tool_records = json.loads(
                        self.manifest_path.read_text(encoding="utf-8")
                    ).get("tools", {})
                except Exception:
                    existing_tool_records = {}

            for name, spec in sorted(STANDALONE_TOOLS.items()):
                if name not in self.skip_tools or name in installed_manifest_tools:
                    continue
                ex_info = existing_tool_records.get(name)
                if not ex_info:
                    continue
                expected_sha = self.sha_overrides.get(name, spec["sha256"])
                if ex_info.get("sha256") != expected_sha or ex_info.get("version") != spec["version"]:
                    die(
                        f"cannot preserve skipped tool {name}: existing manifest entry does not match "
                        "current pinned version/checksum; remove the skip or pass matching overrides"
                    )
                payload_target = self.standalone_dir / name
                wrapper_target = self.bin_dir / spec["target_bin"]
                if not payload_target.is_file() or not wrapper_target.is_file():
                    die(f"cannot preserve skipped tool {name}: existing payload or wrapper file missing")
                rec_payload_sha = ex_info.get("payload_sha256")
                if not rec_payload_sha:
                    die(f"cannot preserve skipped tool {name}: existing manifest has no payload_sha256")
                if compute_sha256(payload_target) != rec_payload_sha:
                    die(f"cannot preserve skipped tool {name}: existing payload checksum mismatch")
                staged_payload = staging_standalone / name
                shutil.copyfile(payload_target, staged_payload)
                staged_payload.chmod(payload_target.stat().st_mode)
                staged_wrapper = staging_bin / spec["target_bin"]
                shutil.copyfile(wrapper_target, staged_wrapper)
                staged_wrapper.chmod(wrapper_target.stat().st_mode)
                preserved = dict(ex_info)
                preserved["installed_sha256"] = compute_sha256(staged_wrapper)
                preserved["payload_sha256"] = rec_payload_sha
                installed_manifest_tools[name] = preserved

            # Preserve the bunx companion wrapper when bunx is skipped but its existing record
            # matches the pinned bun checksum and the bun payload remains installed.
            if "bunx" in self.skip_tools and "bunx" not in installed_manifest_tools:
                ex_bunx = existing_tool_records.get("bunx")
                if ex_bunx and "bun" in installed_manifest_tools:
                    expected_bunx_sha = self.sha_overrides.get("bun", PINNED_BUN_SHA256)
                    if ex_bunx.get("sha256") == expected_bunx_sha and ex_bunx.get("version") == PINNED_BUN_VERSION:
                        bunx_target = self.bin_dir / "bunx"
                        if bunx_target.is_file():
                            staged_bunx = staging_bin / "bunx"
                            shutil.copyfile(bunx_target, staged_bunx)
                            staged_bunx.chmod(bunx_target.stat().st_mode)
                            preserved_bunx = dict(ex_bunx)
                            preserved_bunx["installed_sha256"] = compute_sha256(staged_bunx)
                            installed_manifest_tools["bunx"] = preserved_bunx

            # 2. Extract Alpine package closures (runtime alpine-root and build alpine-build-root)
            # Shared download cache to avoid redownloading common packages across roots
            downloaded_cache: dict[str, Path] = {}

            pkg_closure = [
                (pkg, sha, lic)
                for (pkg, sha, lic) in ALPINE_PACKAGE_CLOSURE
                if (not self.mock_closure) or (pkg in self.artifact_overrides)
            ]

            for pkg_name, default_sha, _ in pkg_closure:
                expected_sha = self.sha_overrides.get(pkg_name, default_sha)
                if pkg_name in self.artifact_overrides:
                    apk_file = self.artifact_overrides[pkg_name]
                    if not apk_file.is_file():
                        die(f"override package missing: {apk_file}")
                elif pkg_name in downloaded_cache:
                    apk_file = downloaded_cache[pkg_name]
                else:
                    raw_url = self.url_overrides.get(pkg_name, f"{ALPINE_DL_BASE}/{pkg_name}")
                    download_path = staging_dir / f"download-{pkg_name}"
                    sys.stderr.write(f"Downloading {pkg_name}...\n")
                    try:
                        urllib.request.urlretrieve(raw_url, download_path)
                        apk_file = download_path
                        downloaded_cache[pkg_name] = download_path
                    except Exception as exc:
                        die(f"download failed for {pkg_name}: {sanitize_text(str(exc))}")

                actual_sha = compute_sha256(apk_file)
                if actual_sha != expected_sha:
                    die(f"checksum mismatch for package {pkg_name}: expected {expected_sha}, got {actual_sha}")

                self._safe_extract_apk(apk_file, staging_alpine_root)

            # Extract 58-package build closure into staging_alpine_build_root
            build_pkg_closure = [
                (pkg, sha, lic, repo)
                for (pkg, sha, lic, repo) in ALPINE_BUILD_PACKAGE_CLOSURE
                if (not self.mock_closure) or (pkg in self.artifact_overrides)
            ]

            for pkg_name, default_sha, _, repo in build_pkg_closure:
                expected_sha = self.sha_overrides.get(pkg_name, default_sha)
                if pkg_name in self.artifact_overrides:
                    apk_file = self.artifact_overrides[pkg_name]
                    if not apk_file.is_file():
                        die(f"override package missing: {apk_file}")
                elif pkg_name in downloaded_cache:
                    apk_file = downloaded_cache[pkg_name]
                else:
                    base_url = ALPINE_COMMUNITY_DL_BASE if repo == "community" else ALPINE_DL_BASE
                    raw_url = self.url_overrides.get(pkg_name, f"{base_url}/{pkg_name}")
                    download_path = staging_dir / f"download-build-{pkg_name}"
                    sys.stderr.write(f"Downloading build package {pkg_name}...\n")
                    try:
                        urllib.request.urlretrieve(raw_url, download_path)
                        apk_file = download_path
                        downloaded_cache[pkg_name] = download_path
                    except Exception as exc:
                        die(f"download failed for build package {pkg_name}: {sanitize_text(str(exc))}")

                actual_sha = compute_sha256(apk_file)
                if actual_sha != expected_sha:
                    die(f"checksum mismatch for build package {pkg_name}: expected {expected_sha}, got {actual_sha}")

                self._safe_extract_apk(apk_file, staging_alpine_build_root)

            # 3. Create wrappers for Alpine tools
            for tool_name, spec in sorted(ALPINE_TOOL_SPECS.items()):
                if tool_name in self.skip_tools:
                    continue
                wrapper_target = staging_bin / spec["target_bin"]
                self._create_alpine_wrapper(tool_name, spec, wrapper_target)
                installed_manifest_tools[tool_name] = dict(tools[tool_name])
                installed_manifest_tools[tool_name]["installed_sha256"] = compute_sha256(wrapper_target)

            # 4. Write manifest
            final_manifest = dict(manifest)
            final_manifest["tools"] = installed_manifest_tools
            manifest_json_bytes = json.dumps(final_manifest, indent=2, sort_keys=True).encode("utf-8")
            (staging_toolchain / "bootstrap-manifest.json").write_bytes(manifest_json_bytes)

            # 5. Atomic same-filesystem transactional replacement
            self.bin_dir.mkdir(parents=True, exist_ok=True)

            backup_alpine_root = None
            backup_alpine_build_root = None
            backup_python_root = None
            backup_standalone = None
            backup_manifest = None
            backup_wrappers: dict[Path, bytes] = {}

            # Detect if current executing Python process is running from inside self.python_root
            executing_py = Path(os.environ.get("PORTFOLIO_LAB_SIMULATE_ACTIVE_PYTHON") or sys.executable).resolve()
            resolved_py_root = self.python_root.resolve()
            active_python_in_tree = False
            try:
                active_python_in_tree = executing_py.is_relative_to(resolved_py_root)
            except Exception:
                active_python_in_tree = False

            try:
                # 5a. Backup existing manifest and wrappers
                if self.manifest_path.is_file():
                    backup_manifest = self.manifest_path.read_bytes()

                for f in staging_bin.iterdir():
                    dest = self.bin_dir / f.name
                    if dest.is_file():
                        backup_wrappers[dest] = dest.read_bytes()

                # 5b. Backup and replace alpine-root
                if self.alpine_root.exists():
                    backup_alpine_root = self.toolchain_dir / ".alpine-root.old"
                    if backup_alpine_root.exists():
                        shutil.rmtree(backup_alpine_root)
                    os.replace(self.alpine_root, backup_alpine_root)
                os.replace(staging_alpine_root, self.alpine_root)

                # 5b2. Backup and replace alpine-build-root
                if self.alpine_build_root.exists():
                    backup_alpine_build_root = self.toolchain_dir / ".alpine-build-root.old"
                    if backup_alpine_build_root.exists():
                        shutil.rmtree(backup_alpine_build_root)
                    os.replace(self.alpine_build_root, backup_alpine_build_root)
                os.replace(staging_alpine_build_root, self.alpine_build_root)

                # 5c. Backup and replace python-root
                if (staging_python_root / "bin" / "python3").is_file():
                    if self.python_root.exists():
                        backup_python_root = self.toolchain_dir / ".python-root.old"
                        if backup_python_root.exists():
                            # If executing python is in .python-root.old, we cannot rmtree it directly;
                            # move it to a trash name or avoid deleting active files.
                            shutil.rmtree(backup_python_root, ignore_errors=True)
                        os.replace(self.python_root, backup_python_root)
                    os.replace(staging_python_root, self.python_root)

                # 5d. Backup and replace standalone
                if (staging_standalone / "bun").is_file() or (staging_standalone / "uv").is_file() or (staging_standalone / "ninja").is_file():
                    if self.standalone_dir.exists():
                        backup_standalone = self.toolchain_dir / ".standalone.old"
                        if backup_standalone.exists():
                            shutil.rmtree(backup_standalone)
                        os.replace(self.standalone_dir, backup_standalone)
                    os.replace(staging_standalone, self.standalone_dir)

                # 5e. Publish wrappers atomically via temporary files on same filesystem
                for f in staging_bin.iterdir():
                    dest = self.bin_dir / f.name
                    tmp_dest = self.bin_dir / f".{f.name}.new"
                    shutil.copyfile(f, tmp_dest)
                    tmp_dest.chmod(f.stat().st_mode)
                    os.replace(tmp_dest, dest)

                # Failure injection point for regression test
                if os.environ.get("PORTFOLIO_LAB_INJECT_INSTALL_FAILURE") == "late_manifest_failure":
                    raise RuntimeError("Injected late install failure before manifest publication")

                # 5f. Publish manifest atomically via temporary file
                tmp_manifest = self.manifest_path.parent / ".bootstrap-manifest.json.new"
                tmp_manifest.write_bytes(manifest_json_bytes)
                os.replace(tmp_manifest, self.manifest_path)

            except Exception as exc:
                # ROLLBACK all changes on any exception
                sys.stderr.write(f"Install failed; rolling back transactional changes: {exc}\n")

                # Restore manifest
                if backup_manifest is not None:
                    self.manifest_path.write_bytes(backup_manifest)
                elif self.manifest_path.exists():
                    self.manifest_path.unlink()

                # Restore wrappers
                for wrapper_dest, content in backup_wrappers.items():
                    wrapper_dest.write_bytes(content)
                for f in staging_bin.iterdir():
                    dest = self.bin_dir / f.name
                    if dest not in backup_wrappers and dest.exists():
                        dest.unlink()

                # Restore directories
                if backup_standalone is not None and backup_standalone.exists():
                    if self.standalone_dir.exists():
                        shutil.rmtree(self.standalone_dir, ignore_errors=True)
                    os.replace(backup_standalone, self.standalone_dir)
                elif self.standalone_dir.exists() and not backup_standalone:
                    shutil.rmtree(self.standalone_dir, ignore_errors=True)

                if backup_python_root is not None and backup_python_root.exists():
                    if self.python_root.exists():
                        shutil.rmtree(self.python_root, ignore_errors=True)
                    os.replace(backup_python_root, self.python_root)
                elif self.python_root.exists() and not backup_python_root:
                    shutil.rmtree(self.python_root, ignore_errors=True)

                if backup_alpine_root is not None and backup_alpine_root.exists():
                    if self.alpine_root.exists():
                        shutil.rmtree(self.alpine_root, ignore_errors=True)
                    os.replace(backup_alpine_root, self.alpine_root)
                elif self.alpine_root.exists() and not backup_alpine_root:
                    shutil.rmtree(self.alpine_root, ignore_errors=True)

                if backup_alpine_build_root is not None and backup_alpine_build_root.exists():
                    if self.alpine_build_root.exists():
                        shutil.rmtree(self.alpine_build_root, ignore_errors=True)
                    os.replace(backup_alpine_build_root, self.alpine_build_root)
                elif self.alpine_build_root.exists() and not backup_alpine_build_root:
                    shutil.rmtree(self.alpine_build_root, ignore_errors=True)

                raise exc

            # Cleanup backups only after successful completion
            if backup_alpine_root and backup_alpine_root.exists():
                shutil.rmtree(backup_alpine_root, ignore_errors=True)
            if backup_alpine_build_root and backup_alpine_build_root.exists():
                shutil.rmtree(backup_alpine_build_root, ignore_errors=True)
            if backup_standalone and backup_standalone.exists():
                shutil.rmtree(backup_standalone, ignore_errors=True)
            if backup_python_root and backup_python_root.exists():
                # Never delete backup_python_root if the current interpreter was invoked from the old tree!
                if not active_python_in_tree:
                    shutil.rmtree(backup_python_root, ignore_errors=True)

            print(f"Successfully installed toolchain to {self.prefix}")

        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def verify(self) -> None:
        if not self.manifest_path.is_file():
            die(f"bootstrap manifest missing: {self.manifest_path}")

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        tools = manifest.get("tools", {})
        packages = manifest.get("packages", {})

        # Reject schema/layout mismatch before behavioral checks
        if manifest.get("schema_version") != SCHEMA_VERSION:
            die(
                f"bootstrap manifest schema mismatch: expected {SCHEMA_VERSION}, got {manifest.get('schema_version')}"
            )
        if manifest.get("layout_revision") != LAYOUT_REVISION:
            die(
                f"bootstrap manifest layout mismatch: expected {LAYOUT_REVISION}, got {manifest.get('layout_revision')}"
            )

        # Verify package artifact records in manifest
        for pkg_name, pkg_info in packages.items():
            if not pkg_info.get("sha256") or not pkg_info.get("source_url"):
                die(f"package manifest record incomplete for {pkg_name}")

        # Check managed root directories exist
        if not self.bin_dir.is_dir():
            die(f"bin directory missing: {self.bin_dir}")
        if not self.alpine_root.is_dir():
            die(f"alpine root directory missing: {self.alpine_root}")
        if not self.alpine_build_root.is_dir():
            die(f"alpine build root directory missing: {self.alpine_build_root}")

        # Validate build_packages manifest records against expected closure metadata
        expected_build_closure: dict[str, tuple[str, str, str]] = {
            pkg: (sha, lic, repo) for (pkg, sha, lic, repo) in ALPINE_BUILD_PACKAGE_CLOSURE
        }
        build_packages = manifest.get("build_packages", {})

        if not self.mock_closure:
            if set(build_packages.keys()) != set(expected_build_closure.keys()):
                die(f"build_packages closure count/set mismatch: expected {len(expected_build_closure)} packages, got {len(build_packages)}")

        # Validate every tracked build_package record against expected metadata
        for pkg_name, rec in build_packages.items():
            if pkg_name not in expected_build_closure:
                die(f"unexpected build package record in manifest: {pkg_name}")
            def_sha, def_lic, def_repo = expected_build_closure[pkg_name]
            base_url = ALPINE_COMMUNITY_DL_BASE if def_repo == "community" else ALPINE_DL_BASE
            def_url = f"{base_url}/{pkg_name}"

            exp_sha = self.sha_overrides.get(pkg_name, def_sha)
            exp_lic = def_lic
            exp_repo = def_repo
            exp_url = self.url_overrides.get(pkg_name, def_url)

            # sha256: mock fixture records are authoritative for the checksum (presence enforced);
            # production must match the pinned/override checksum exactly.
            if self.mock_closure:
                if not rec.get("sha256"):
                    die(f"build package {pkg_name} missing sha256 in manifest record")
            elif rec.get("sha256") != exp_sha:
                die(f"build package {pkg_name} sha256 mismatch: expected {exp_sha}, got {rec.get('sha256')}")

            # Structural metadata validates identically in both modes; mock mode must never
            # accept arbitrary nonempty values for license/repository/install_root/source_url.
            if rec.get("license") != exp_lic:
                die(f"build package {pkg_name} license mismatch: expected {exp_lic}, got {rec.get('license')}")
            if rec.get("repository") != exp_repo:
                die(f"build package {pkg_name} repository mismatch: expected {exp_repo}, got {rec.get('repository')}")
            if rec.get("install_root") != str(self.alpine_build_root):
                die(f"build package {pkg_name} install_root mismatch: expected {self.alpine_build_root}, got {rec.get('install_root')}")
            if pkg_name in self.url_overrides:
                if rec.get("source_url") != exp_url and rec.get("source_url") != redact_url(exp_url):
                    die(f"build package {pkg_name} source_url mismatch: expected {exp_url}, got {rec.get('source_url')}")
            elif not is_acceptable_manifest_url(rec.get("source_url", "")):
                die(f"build package {pkg_name} source_url invalid or unsafe: {rec.get('source_url')}")

        # Validate runtime packages manifest records against expected closure metadata
        expected_pkg_closure: dict[str, tuple[str, str, str]] = {
            pkg: (sha, lic, "main") for (pkg, sha, lic) in ALPINE_PACKAGE_CLOSURE
        }
        packages = manifest.get("packages", {})

        if not self.mock_closure:
            if set(packages.keys()) != set(expected_pkg_closure.keys()):
                die(f"packages closure count/set mismatch: expected {len(expected_pkg_closure)} packages, got {len(packages)}")

        for pkg_name, rec in packages.items():
            if pkg_name not in expected_pkg_closure:
                die(f"unexpected runtime package record in manifest: {pkg_name}")
            def_sha, def_lic, def_repo = expected_pkg_closure[pkg_name]
            base_url = ALPINE_COMMUNITY_DL_BASE if def_repo == "community" else ALPINE_DL_BASE
            def_url = f"{base_url}/{pkg_name}"

            exp_sha = self.sha_overrides.get(pkg_name, def_sha)
            exp_lic = def_lic
            exp_repo = def_repo
            exp_url = self.url_overrides.get(pkg_name, def_url)

            # Runtime package records: sha256 fixtures are mock-authoritative (presence enforced),
            # production must match pins exactly; structural metadata validates identically in both
            # modes so mock mode never accepts arbitrary nonempty values.
            if self.mock_closure:
                if not rec.get("sha256"):
                    die(f"runtime package {pkg_name} missing sha256 in manifest record")
            elif rec.get("sha256") != exp_sha:
                die(f"runtime package {pkg_name} sha256 mismatch: expected {exp_sha}, got {rec.get('sha256')}")
            if rec.get("license") != exp_lic:
                die(f"runtime package {pkg_name} license mismatch: expected {exp_lic}, got {rec.get('license')}")
            if rec.get("repository") != exp_repo:
                die(f"runtime package {pkg_name} repository mismatch: expected {exp_repo}, got {rec.get('repository')}")
            if rec.get("install_root") != str(self.alpine_root):
                die(f"runtime package {pkg_name} install_root mismatch: expected {self.alpine_root}, got {rec.get('install_root')}")
            if pkg_name in self.url_overrides:
                if rec.get("source_url") != exp_url and rec.get("source_url") != redact_url(exp_url):
                    die(f"runtime package {pkg_name} source_url mismatch: expected {exp_url}, got {rec.get('source_url')}")
            elif not is_acceptable_manifest_url(rec.get("source_url", "")):
                die(f"runtime package {pkg_name} source_url invalid or unsafe: {rec.get('source_url')}")

        # Check CA bundle structurally exists under alpine-root
        ca_bundle_path = self.alpine_root / "etc" / "ssl" / "certs" / "ca-certificates.crt"
        if not ca_bundle_path.is_file():
            die(f"CA bundle missing under alpine-root: {ca_bundle_path}")

        # Create guaranteed cleanup test directory under toolchain
        test_scratch = Path(tempfile.mkdtemp(prefix=".verify-scratch-", dir=self.toolchain_dir))
        timeout_sec = 15
        try:
            for name, info in sorted(tools.items()):
                if name in self.skip_tools:
                    continue

                target = Path(info["install_path"])
                if not target.is_file():
                    die(f"tool binary missing: {target}")

                recorded_sha = info.get("installed_sha256")
                if recorded_sha:
                    current_sha = compute_sha256(target)
                    if current_sha != recorded_sha:
                        die(f"tampered binary detected at {target}: expected {recorded_sha}, got {current_sha}")

                # Verify recorded payload/runtime hashes for standalone tools
                if name in ("bun", "uv", "ninja"):
                    payload_target = self.standalone_dir / name
                    if not payload_target.is_file():
                        die(f"standalone payload missing for {name}: {payload_target}")
                    rec_payload_sha = info.get("payload_sha256")
                    if not rec_payload_sha:
                        die(f"missing recorded payload_sha256 in manifest for {name}")
                    cur_payload_sha = compute_sha256(payload_target)
                    if cur_payload_sha != rec_payload_sha:
                        die(f"tampered standalone payload detected for {name}: expected {rec_payload_sha}, got {cur_payload_sha}")
                elif name == "python3":
                    py_runtime_target = self.python_root / "bin" / "python3"
                    if not py_runtime_target.is_file():
                        die(f"python runtime binary missing: {py_runtime_target}")
                    rec_runtime_sha = info.get("runtime_sha256")
                    if not rec_runtime_sha:
                        die("missing recorded runtime_sha256 in manifest for python3")
                    cur_runtime_sha = compute_sha256(py_runtime_target)
                    if cur_runtime_sha != rec_runtime_sha:
                        die(f"tampered python runtime binary detected: expected {rec_runtime_sha}, got {cur_runtime_sha}")

                # Verify behavioral execution
                if name == "python3":
                    # Python must import encodings, sqlite3, os and execute an in-memory SQLite query
                    py_code = (
                        "import encodings, sqlite3, os; "
                        "con = sqlite3.connect(':memory:'); "
                        "cur = con.cursor(); "
                        "cur.execute('select 42, 1'); "
                        "row = cur.fetchone(); "
                        "assert row == (42, 1), f'unexpected row: {row}'; "
                        "con.close(); "
                        "print('PY_BEHAVIOR_OK')"
                    )
                    try:
                        res = subprocess.run(
                            [str(target), "-c", py_code],
                            capture_output=True,
                            text=True,
                            timeout=timeout_sec,
                        )
                    except subprocess.TimeoutExpired:
                        die("python3 behavioral verify timed out")
                    if res.returncode != 0 or "PY_BEHAVIOR_OK" not in res.stdout:
                        die(f"python3 runtime verify failed: {res.stderr.strip() or res.stdout.strip()}")

                elif name == "git":
                    # Git must initialize a repo with local config, commit, bundle, verify bundle, and clone
                    git_repo = test_scratch / "git-test-repo"
                    git_clone = test_scratch / "git-test-clone"
                    bundle_file = test_scratch / "repo.bundle"
                    try:
                        run_cmd = lambda cmd, cwd=None: subprocess.run(  # noqa: E731
                            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_sec
                        )
                        r = run_cmd([str(target), "init", str(git_repo)])
                        if r.returncode != 0:
                            die(f"git verify failed (init): {r.stderr.strip()}")

                        r = run_cmd([str(target), "config", "user.email", "ci@portfolio-lab.local"], cwd=git_repo)
                        if r.returncode != 0:
                            die(f"git verify failed (config user.email): {r.stderr.strip()}")

                        r = run_cmd([str(target), "config", "user.name", "Portfolio Lab CI"], cwd=git_repo)
                        if r.returncode != 0:
                            die(f"git verify failed (config user.name): {r.stderr.strip()}")

                        test_file = git_repo / "test.txt"
                        test_file.write_text("verification_commit_payload\n", encoding="utf-8")
                        r = run_cmd([str(target), "add", "test.txt"], cwd=git_repo)
                        if r.returncode != 0:
                            die(f"git verify failed (add): {r.stderr.strip()}")

                        r = run_cmd([str(target), "commit", "-m", "init commit"], cwd=git_repo)
                        if r.returncode != 0:
                            die(f"git verify failed (commit): {r.stderr.strip()}")

                        r = run_cmd([str(target), "bundle", "create", str(bundle_file), "HEAD"], cwd=git_repo)
                        if r.returncode != 0:
                            die(f"git verify failed (bundle create): {r.stderr.strip()}")

                        r = run_cmd([str(target), "bundle", "verify", str(bundle_file)], cwd=git_repo)
                        if r.returncode != 0:
                            die(f"git verify failed (bundle verify): {r.stderr.strip()}")

                        r = run_cmd([str(target), "clone", str(bundle_file), str(git_clone)])
                        if r.returncode != 0:
                            die(f"git verify failed (clone bundle): {r.stderr.strip()}")

                        cloned_content = (git_clone / "test.txt").read_text(encoding="utf-8")
                        if "verification_commit_payload" not in cloned_content:
                            die("git verify failed: cloned file content mismatch")
                    except subprocess.TimeoutExpired:
                        die("git behavioral verify timed out")

                elif name == "curl":
                    # curl verifies version and private CA config; optional bounded HTTPS request
                    wrapper_content = target.read_text(encoding="utf-8")
                    if "SSL_CERT_FILE" not in wrapper_content or "CURL_CA_BUNDLE" not in wrapper_content:
                        die("curl wrapper missing private CA environment exports")

                    res = subprocess.run([str(target), "--version"], capture_output=True, text=True, timeout=timeout_sec)
                    if res.returncode != 0 or "curl" not in res.stdout:
                        die(f"curl verify failed: {res.stderr.strip()}")

                    if self.network_verify:
                        res_net = subprocess.run(
                            [str(target), "-sS", "--max-time", "10", "-I", "https://dl-cdn.alpinelinux.org/"],
                            capture_output=True,
                            text=True,
                            timeout=timeout_sec,
                        )
                        if res_net.returncode != 0:
                            die(f"curl network verify failed: {res_net.stderr.strip()}")

                elif name == "sqlite3":
                    # sqlite3 executes :memory: with select 1; pragma integrity_check;
                    query = "select 1; pragma integrity_check;"
                    try:
                        res = subprocess.run(
                            [str(target), ":memory:", query],
                            capture_output=True,
                            text=True,
                            timeout=timeout_sec,
                        )
                    except subprocess.TimeoutExpired:
                        die("sqlite3 behavioral verify timed out")
                    if res.returncode != 0 or "ok" not in res.stdout.lower() or "1" not in res.stdout:
                        die(f"sqlite3 verify failed: {res.stderr.strip() or res.stdout.strip()}")

                elif name == "rsync":
                    # rsync copies a file locally and verifies content
                    src_file = test_scratch / "rsync_src.txt"
                    dst_file = test_scratch / "rsync_dst.txt"
                    src_file.write_text("rsync_verification_token\n", encoding="utf-8")
                    try:
                        res = subprocess.run(
                            [str(target), str(src_file), str(dst_file)],
                            capture_output=True,
                            text=True,
                            timeout=timeout_sec,
                        )
                    except subprocess.TimeoutExpired:
                        die("rsync behavioral verify timed out")
                    if res.returncode != 0 or not dst_file.is_file() or dst_file.read_text(encoding="utf-8") != src_file.read_text(encoding="utf-8"):
                        die(f"rsync verify failed: {res.stderr.strip()}")

                elif name == "jq":
                    # jq parses JSON and returns asserted value
                    try:
                        p = subprocess.Popen(
                            [str(target), ".foo"],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                        )
                        out, err = p.communicate(input='{"foo": "jq_verification_ok"}', timeout=timeout_sec)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        die("jq behavioral verify timed out")
                    if p.returncode != 0 or "jq_verification_ok" not in out:
                        die(f"jq verify failed: {err.strip() or out.strip()}")

                elif name == "zstd":
                    # zstd compresses and decompresses payload
                    z_payload = test_scratch / "zstd_payload.txt"
                    z_compressed = test_scratch / "zstd_payload.txt.zst"
                    z_decompressed = test_scratch / "zstd_restored.txt"
                    raw_data = "zstd_verification_payload_42\n"
                    z_payload.write_text(raw_data, encoding="utf-8")
                    try:
                        r1 = subprocess.run([str(target), "-q", str(z_payload), "-o", str(z_compressed)], capture_output=True, text=True, timeout=timeout_sec)
                        if r1.returncode != 0:
                            die(f"zstd verify failed (compression): {r1.stderr.strip()}")
                        r2 = subprocess.run([str(target), "-d", "-q", str(z_compressed), "-o", str(z_decompressed)], capture_output=True, text=True, timeout=timeout_sec)
                        if r2.returncode != 0:
                            die(f"zstd verify failed (decompression): {r2.stderr.strip()}")
                        if z_decompressed.read_text(encoding="utf-8") != raw_data:
                            die("zstd verify failed: decompressed payload mismatch")
                    except subprocess.TimeoutExpired:
                        die("zstd behavioral verify timed out")

                elif name == "bun":
                    res = subprocess.run([str(target), "--version"], capture_output=True, text=True, timeout=timeout_sec)
                    if res.returncode != 0 or "1.4" not in res.stdout:
                        die(f"bun verify failed: {res.stderr.strip() or res.stdout.strip()}")

                elif name == "bunx":
                    # bunx wrapper delegates to bun x "$@"; test with arguments without network
                    try:
                        res = subprocess.run(
                            [str(target), "--bun", "--version"],
                            capture_output=True,
                            text=True,
                            timeout=timeout_sec,
                        )
                    except subprocess.TimeoutExpired:
                        die("bunx behavioral verify timed out")
                    if res.returncode != 0:
                        die(f"bunx verify failed: {res.stderr.strip() or res.stdout.strip()}")

                elif name == "uv":
                    res = subprocess.run([str(target), "--version"], capture_output=True, text=True, timeout=timeout_sec)
                    if res.returncode != 0 or "uv" not in res.stdout:
                        die(f"uv verify failed: {res.stderr.strip()}")

                elif name == "ninja":
                    res = subprocess.run([str(target), "--version"], capture_output=True, text=True, cwd=test_scratch, timeout=timeout_sec)
                    if res.returncode != 0 or "1.13" not in res.stdout:
                        die(f"ninja verify failed: {res.stderr.strip() or res.stdout.strip()}")

                    # Behavioral build probe using real Ninja syntax
                    ninja_probe_file = test_scratch / "verify_ninja.build"
                    ninja_probe_out = test_scratch / "ninja_probe.txt"
                    ninja_probe_file.write_text(
                        f"rule echo_rule\n  command = echo NINJA_PROBE_OK > $out\n\nbuild {ninja_probe_out}: echo_rule\n",
                        encoding="utf-8",
                    )
                    try:
                        res_ninja = subprocess.run(
                            [str(target), "-f", str(ninja_probe_file)],
                            capture_output=True,
                            text=True,
                            cwd=test_scratch,
                            timeout=timeout_sec,
                        )
                    except subprocess.TimeoutExpired:
                        die("ninja behavioral verify timed out")
                    if res_ninja.returncode != 0 or not ninja_probe_out.is_file() or "NINJA_PROBE_OK" not in ninja_probe_out.read_text(encoding="utf-8"):
                        die(f"ninja behavioral execution failed: {res_ninja.stderr.strip() or res_ninja.stdout.strip()}")

            # Verify behavioral build tools and compilers from managed build-root
            # In mock mode, only run build tool probes if build tools were installed in build-root
            build_bin_dir = self.alpine_build_root / "usr" / "bin"
            should_check_build_tools = (not self.mock_closure) or (build_bin_dir / "gcc").is_file()

            if should_check_build_tools:
                build_env = self.get_build_environment()

                build_tools_to_check = [
                    ("gcc", ["--version"], "gcc"),
                    ("g++", ["--version"], "g++"),
                    ("gfortran", ["--version"], "fortran"),
                    ("rustc", ["--version"], "rustc"),
                    ("cargo", ["--version"], "cargo"),
                    ("cmake", ["--version"], "cmake"),
                ]

                resolved_build_root = self.alpine_build_root.resolve()

                # Validate tool binary presence, command identity, and containment
                for btool, ver_arg, exp_token in build_tools_to_check:
                    btool_path = build_bin_dir / btool
                    if not btool_path.is_file():
                        die(f"managed build tool missing in alpine-build-root: {btool_path}")

                    # Containment check: tool must not resolve outside alpine-build-root
                    resolved_tool = btool_path.resolve()
                    if not (resolved_tool == resolved_build_root or resolved_tool.is_relative_to(resolved_build_root)):
                        die(f"managed build tool path escapes alpine-build-root: {btool_path} -> {resolved_tool}")

                    try:
                        r_tool = subprocess.run(
                            [str(btool_path), *ver_arg],
                            capture_output=True,
                            text=True,
                            env=build_env,
                            timeout=timeout_sec,
                        )
                    except subprocess.TimeoutExpired:
                        die(f"{btool} verify timed out")
                    if r_tool.returncode != 0 or exp_token not in r_tool.stdout.lower():
                        die(f"{btool} verify failed: {r_tool.stderr.strip() or r_tool.stdout.strip()}")

                # pkg-config or pkgconf presence, containment, and command identity
                pkgconf_path = build_bin_dir / "pkgconf"
                pkgconfig_path = build_bin_dir / "pkg-config"
                if not pkgconf_path.is_file() and not pkgconfig_path.is_file():
                    die(f"pkg-config/pkgconf missing in alpine-build-root: neither {pkgconf_path} nor {pkgconfig_path} found")
                active_pkgconf = pkgconf_path if pkgconf_path.is_file() else pkgconfig_path

                resolved_pkgconf = active_pkgconf.resolve()
                if not (resolved_pkgconf == resolved_build_root or resolved_pkgconf.is_relative_to(resolved_build_root)):
                    die(f"pkg-config/pkgconf path escapes alpine-build-root: {active_pkgconf} -> {resolved_pkgconf}")

                try:
                    r_pkg = subprocess.run(
                        [str(active_pkgconf), "--version"],
                        capture_output=True,
                        text=True,
                        env=build_env,
                        timeout=timeout_sec,
                    )
                except subprocess.TimeoutExpired:
                    die("pkg-config/pkgconf verify timed out")
                if r_pkg.returncode != 0:
                    die(f"pkg-config/pkgconf verify failed: {r_pkg.stderr.strip() or r_pkg.stdout.strip()}")
                pkg_ver = r_pkg.stdout.strip()
                if not pkg_ver or not re.search(r"\d+\.\d+", pkg_ver):
                    die(f"pkg-config/pkgconf returned invalid version output: '{pkg_ver}'")

                # Bounded compile/link/run probe for C
                c_src = test_scratch / "probe.c"
                c_bin = test_scratch / "probe_c_out"
                c_src.write_text("int main(void) { return 0; }\n", encoding="utf-8")
                try:
                    r_c_comp = subprocess.run(
                        [str(build_bin_dir / "gcc"), str(c_src), "-o", str(c_bin)],
                        capture_output=True,
                        text=True,
                        env=build_env,
                        timeout=timeout_sec,
                    )
                except subprocess.TimeoutExpired:
                    die("gcc compile probe timed out")
                if r_c_comp.returncode != 0 or not c_bin.is_file():
                    die(f"gcc compile probe failed: {r_c_comp.stderr.strip() or r_c_comp.stdout.strip()}")
                try:
                    r_c_run = subprocess.run([str(c_bin)], capture_output=True, text=True, env=build_env, timeout=timeout_sec)
                except subprocess.TimeoutExpired:
                    die("gcc executable run probe timed out")
                if r_c_run.returncode != 0:
                    die(f"gcc run probe failed: {r_c_run.stderr.strip() or r_c_run.stdout.strip()}")

                # Bounded compile/link/run probe for C++
                cpp_src = test_scratch / "probe.cpp"
                cpp_bin = test_scratch / "probe_cpp_out"
                cpp_src.write_text("int main() { return 0; }\n", encoding="utf-8")
                try:
                    r_cpp_comp = subprocess.run(
                        [str(build_bin_dir / "g++"), str(cpp_src), "-o", str(cpp_bin)],
                        capture_output=True,
                        text=True,
                        env=build_env,
                        timeout=timeout_sec,
                    )
                except subprocess.TimeoutExpired:
                    die("g++ compile probe timed out")
                if r_cpp_comp.returncode != 0 or not cpp_bin.is_file():
                    die(f"g++ compile probe failed: {r_cpp_comp.stderr.strip() or r_cpp_comp.stdout.strip()}")
                try:
                    r_cpp_run = subprocess.run([str(cpp_bin)], capture_output=True, text=True, env=build_env, timeout=timeout_sec)
                except subprocess.TimeoutExpired:
                    die("g++ executable run probe timed out")
                if r_cpp_run.returncode != 0:
                    die(f"g++ run probe failed: {r_cpp_run.stderr.strip() or r_cpp_run.stdout.strip()}")

                # Bounded compile/link/run probe for Fortran
                f_src = test_scratch / "probe.f90"
                f_bin = test_scratch / "probe_f_out"
                f_src.write_text("program test\nend program test\n", encoding="utf-8")
                try:
                    r_f_comp = subprocess.run(
                        [str(build_bin_dir / "gfortran"), str(f_src), "-o", str(f_bin)],
                        capture_output=True,
                        text=True,
                        env=build_env,
                        timeout=timeout_sec,
                    )
                except subprocess.TimeoutExpired:
                    die("gfortran compile probe timed out")
                if r_f_comp.returncode != 0 or not f_bin.is_file():
                    die(f"gfortran compile probe failed: {r_f_comp.stderr.strip() or r_f_comp.stdout.strip()}")
                try:
                    r_f_run = subprocess.run([str(f_bin)], capture_output=True, text=True, env=build_env, timeout=timeout_sec)
                except subprocess.TimeoutExpired:
                    die("gfortran executable run probe timed out")
                if r_f_run.returncode != 0:
                    die(f"gfortran run probe failed: {r_f_run.stderr.strip() or r_f_run.stdout.strip()}")

                # Bounded compile/link/run probe for Rust
                rs_src = test_scratch / "probe.rs"
                rs_bin = test_scratch / "probe_rs_out"
                rs_src.write_text("fn main() {}\n", encoding="utf-8")
                try:
                    r_rs_comp = subprocess.run(
                        [str(build_bin_dir / "rustc"), str(rs_src), "-o", str(rs_bin)],
                        capture_output=True,
                        text=True,
                        env=build_env,
                        timeout=timeout_sec,
                    )
                except subprocess.TimeoutExpired:
                    die("rustc compile probe timed out")
                if r_rs_comp.returncode != 0 or not rs_bin.is_file():
                    die(f"rustc compile probe failed: {r_rs_comp.stderr.strip() or r_rs_comp.stdout.strip()}")
                try:
                    r_rs_run = subprocess.run([str(rs_bin)], capture_output=True, text=True, env=build_env, timeout=timeout_sec)
                except subprocess.TimeoutExpired:
                    die("rustc executable run probe timed out")
                if r_rs_run.returncode != 0:
                    die(f"rustc run probe failed: {r_rs_run.stderr.strip() or r_rs_run.stdout.strip()}")

        finally:
            shutil.rmtree(test_scratch, ignore_errors=True)

        print("Toolchain verified successfully.")

    def uninstall(self, *, dry_run: bool = False) -> None:
        to_remove_files: list[Path] = []
        to_remove_dirs: list[Path] = []

        resolved_prefix = self.prefix.resolve()

        if self.manifest_path.is_file():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            for t in manifest.get("tools", {}).values():
                if "install_path" in t:
                    p = Path(t["install_path"])
                    if not (p.resolve() == resolved_prefix or p.resolve().is_relative_to(resolved_prefix)):
                        die(f"uninstall path containment failure: {p} is outside {resolved_prefix}")
                    to_remove_files.append(p)
            to_remove_files.append(self.manifest_path)
            for d_key in ("alpine_root", "alpine_build_root", "python_root", "standalone_dir", "toolchain_dir"):
                if d_key in manifest:
                    d = Path(manifest[d_key])
                    if not (d.resolve() == resolved_prefix or d.resolve().is_relative_to(resolved_prefix)):
                        die(f"uninstall dir containment failure: {d} is outside {resolved_prefix}")
                    to_remove_dirs.append(d)
        else:
            to_remove_files.extend([self.bin_dir / s["target_bin"] for s in STANDALONE_TOOLS.values()])
            to_remove_files.append(self.bin_dir / "bunx")
            to_remove_files.extend([self.bin_dir / s["target_bin"] for s in ALPINE_TOOL_SPECS.values()])
            to_remove_dirs.append(self.alpine_root)
            to_remove_dirs.append(self.alpine_build_root)
            to_remove_dirs.append(self.python_root)
            to_remove_dirs.append(self.standalone_dir)
            to_remove_dirs.append(self.toolchain_dir)

        for p in sorted(set(to_remove_files)):
            if p.is_file():
                if dry_run:
                    print(f"Would remove file: {p}")
                else:
                    p.unlink()
                    print(f"Removed file: {p}")

        for d in sorted(set(to_remove_dirs), reverse=True):
            if d.is_dir():
                if dry_run:
                    print(f"Would remove directory: {d}")
                else:
                    shutil.rmtree(d)
                    print(f"Removed directory: {d}")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Portfolio Lab cursor-box user-owned dependency bootstrap.",
    )
    parser.add_argument(
        "command",
        choices=["dry-run", "install", "verify", "uninstall", "stage-0-info"],
        help="Action to perform",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"Target installation prefix (default: {DEFAULT_PREFIX})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report actions without mutating files (for uninstall)",
    )
    parser.add_argument(
        "--override-url",
        action="append",
        default=[],
        help="Override tool/package source URL (format: name=URL)",
    )
    parser.add_argument(
        "--override-artifact",
        action="append",
        default=[],
        help="Override tool/package archive file path (format: name=PATH)",
    )
    parser.add_argument(
        "--override-sha256",
        action="append",
        default=[],
        help="Override tool/package expected SHA-256 (format: name=SHA256)",
    )
    parser.add_argument(
        "--skip-tools",
        default="",
        help="Comma-separated tools to skip",
    )
    parser.add_argument(
        "--mock-closure",
        action="store_true",
        help="Test mode: only install packages explicitly overridden in test fixtures",
    )
    parser.add_argument(
        "--stage0-python-archive",
        default="",
        help="Path to pre-transferred Python standalone archive for stage-0 initialization",
    )

    parser.add_argument(
        "--network-verify",
        action="store_true",
        help="Perform bounded live network HTTPS check during verify (optional)",
    )

    args, extra = parser.parse_known_args()
    return args, extra


def main() -> None:
    raw_args = sys.argv[1:]
    check_forbidden_commands(raw_args)

    args, extra = parse_args()
    if extra:
        check_forbidden_commands(extra)
        die(f"unrecognized arguments: {' '.join(extra)}")

    if args.command == "stage-0-info":
        print(f"Stage-0 bootstrap info for prefix: {args.prefix}")
        return

    prefix_path = Path(args.prefix)
    check_root_execution(str(prefix_path))

    url_overrides: dict[str, str] = {}
    for item in args.override_url:
        if "=" in item:
            k, v = item.split("=", 1)
            url_overrides[k] = v

    artifact_overrides: dict[str, Path] = {}
    if args.stage0_python_archive and Path(args.stage0_python_archive).is_file():
        artifact_overrides["python3"] = Path(args.stage0_python_archive)

    for item in args.override_artifact:
        if "=" in item:
            k, v = item.split("=", 1)
            artifact_overrides[k] = Path(v)

    sha_overrides: dict[str, str] = {}
    for item in args.override_sha256:
        if "=" in item:
            k, v = item.split("=", 1)
            sha_overrides[k] = v

    skip_tools = {t.strip() for t in args.skip_tools.split(",") if t.strip()}

    manager = BootstrapManager(
        prefix_path,
        url_overrides=url_overrides,
        artifact_overrides=artifact_overrides,
        sha_overrides=sha_overrides,
        skip_tools=skip_tools,
        mock_closure=args.mock_closure,
        network_verify=args.network_verify,
    )

    if args.command == "dry-run":
        manager.dry_run()
    elif args.command == "install":
        manager.install()
    elif args.command == "verify":
        manager.verify()
    elif args.command == "uninstall":
        manager.uninstall(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
