# Vendored 第三方代码：Sesame

[中文（本页）](PROVENANCE.md)

`reference/sesame/` 是 **Sesame 2.0** 的 vendored 副本——Sesame 是美国国家标准与技术研究院
（NIST）与马里兰大学开发的有限差分漂移-扩散–泊松求解器。

| 项 | 值 |
| --- | --- |
| 项目 | Sesame |
| 版本 | 2.0（`sesame/_version.py`） |
| 作者 | Benoit Gaury（NIST / 马里兰大学）、Paul M. Haney（NIST） |
| 上游 | `http://github.com/usnistgov/sesame`（见 `INSTALL.rst` 记载） |
| 文档 | `http://sesame.readthedocs.io/en/latest/`（见 `README.rst` 记载） |
| 许可证 | 3-clause BSD 风格，`LICENSE.rst`，"Copyright 2017 University of Maryland" |
| 资助 | NIST 与马里兰大学，CRA 70NANB14H209（`AUTHORS.rst`） |

## 改动情况

**保留下来的文件没有任何一个被编辑过。** `LICENSE.rst`、`AUTHORS.rst`、`INSTALL.rst`、
`README.rst`、`setup.py`、`setup.cfg`、`npdoc.txt`、`app.py`、`app.spec` 以及整个可 import 的
包目录 `sesame/`，与入库时归档的副本逐字节一致。

发布本仓库时删除了三个不含求解器代码的子树——文档、示例输入脚本、以及 Sesame 自带的 GUI 测试套件：

| 删除的子树 | `.py` 文件数 | 内容 |
| --- | --- | --- |
| `doc/` | 1 | reST 文档与 SVG/PNG 图（原 15 MB 中的 11 MB） |
| `examples/` | 10 | 独立示例脚本，从不被 import |
| `test_suite/` | 9 | Sesame 自己的、面向 GUI 的测试 |

包目录 `sesame/`（22 个 `.py`，含 `ui/` 和未编译的 `mumps/` 绑定）整块保留、不做挑选式裁剪，
这样"我们发布的任何东西都没被改过"这一条声明保持简单、可核验。

## 完整性摘要

```bash
# 本仓库发布的这棵树：24 个 .py
cd reference/sesame && find . -name "*.py" -type f | sort | xargs cat | sha256sum
# fa1698474ee02d89c632761742c122df5e8f3a13cbe0f68fdcf9f8a481f1ee30

# 裁剪前的完整副本（44 个 .py，即 doc/、examples/、test_suite/ 都在）
# b1b093cf389a487b8d249b9b782e6343ca17aeb36272c94054ccca851cca9cb0
```

第二个摘要是"完整副本 → 本裁剪副本"这一来源关系的凭证；在上游副本上于删除之前运行同一条命令即可复现。

## 本项目怎么用它

只使用可脚本化的 API，调用点在
[`src/p5dd/devices.py`](../src/p5dd/devices.py) 的 `sesame_system()`：

```python
sys_ = sesame.Builder(x, input_length="cm", T=300.0)
sys_.add_material({...})
sys_.add_donor(ND, lambda pos: pos < x_j)
sys_.add_acceptor(NA, lambda pos: pos >= x_j)
sys_.contact_type("Ohmic", "Ohmic")
sys_.contact_S(sv, sv, sv, sv)
J = sesame.IVcurve(sys_, voltages, out, verbose=False, tol=1e-9)
```

`sesame` 是在运行时由
[`scripts/case5_sesame_crosscheck.py`](../scripts/case5_sesame_crosscheck.py) 插入 `sys.path` 的；
它从未被安装，也不会被求解器本身 import，所以删掉这个目录，项目其余部分照常运行。
可选的 MUMPS 直接法绑定（`sesame/mumps/`）未编译、未使用——Sesame 回退到 SciPy，
交叉对照就是在这个配置下跑的。

按上面的裁剪之后，交叉对照仍然端到端跑通：`pytest -q` 报告
**93 collected / 93 passed / 0 skipped**；GitHub Actions 在 Python 3.10 / 3.12 / 3.13 上
三个 job 各自同样是 93 / 93 / 0（案例 5 实际执行，无跳过）。
完全删除 `reference/` 会让案例 5 与 4 项测试变成跳过，其余 89 项照常；
跳过一律不计入通过（见 [`reports/acceptance-checklist.md`](../reports/acceptance-checklist.md)）。

## 再分发

Sesame 是 3-clause BSD：源码或二进制形式的再分发都须保留版权声明与许可证全文，
二者在本仓库中均原样保留。如果你重新发布本仓库，请一并保留
`reference/sesame/LICENSE.rst`、`reference/sesame/AUTHORS.rst` 与本页——
并且不要把 Sesame 算作本项目的自有贡献。它在这里的角色是一根外部的量尺；
它所校核的物理写在 `reports/technical-report.md` §1.3。
