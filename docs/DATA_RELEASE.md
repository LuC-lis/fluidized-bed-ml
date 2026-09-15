# 数据发布方案 / Data release

> **状态**：导师已确认本批实验数据**可以公开**（2026-09-15 口头确认，建议在正式发布前留一封邮件记录）。
> 因此 `docs/CONCRETE_EVIDENCE_PLAN.md` 第 2.4 项的障碍已解除。

## 1. 分工：什么放 GitHub，什么放 Zenodo

| 内容 | 体量 | 去哪 | 为什么 |
|---|---|---|---|
| 代码 + 派生表 + 图 + 文档 | 10 MB | **GitHub**（已完成） | 版本控制、可 review、可 CI |
| 压力原始日志 `Pressure_Energy/*.xlsx` | 27 MB | **GitHub 也可以**，或 Zenodo 一并 | 小，且是论文级结论的原始证据 |
| 光纤探针 `20250609 LC/*.pvd` | 3.0 GB | **Zenodo**（单文件上限 50 GB） | GitHub 单文件上限 100 MB / 仓库建议 < 5 GB |
| 振动 `20250609 LMF/*.mat,*.bin` | 629 MB | **Zenodo** | 同上 |
| 大学生海报 / 汇报 PPT / 论文 PDF | 66 MB | **都不放** | PPT 含个人信息；论文 PDF 版权属出版社，只放 DOI 链接 |

Zenodo 支持"一个 DOI 对应多个文件"，所以最终形态是：

* GitHub 仓库 → 代码，README 里放 Zenodo DOI 徽章
* Zenodo record → 原始数据 + `MANIFEST.csv` + `DATASHEET.md` + 同一份代码的归档 zip

## 2. 具体步骤

```bash
# 1) 生成清单、校验和、数据说明（不复制、不上传，只读）
python scripts/09_package_data_release.py

# 2) 如果确实需要打包（会额外占用约 3.8 GB 磁盘）
python scripts/09_package_data_release.py --create-archive --outdir /Volumes/External/fbf-data

# 3) Zenodo
#    - 注册/登录 zenodo.org，用 GitHub 账号登录
#    - New upload → 上传 release/ 下的分卷压缩包（每个 < 50 GB）
#    - Resource type: Dataset；Licence: CC BY 4.0
#    - 填 DATASHEET.md 的内容；作者按实验室要求署名
#    - Publish → 得到 DOI，例如 10.5281/zenodo.XXXXXXX
#    - 把 DOI 填回 README.md 顶部和 docs/data_dictionary.md
```

## 3. 发布前必须补的三件事

1. **许可与署名**：原始数据由实验室采集，作者顺序与许可（建议 CC BY 4.0）需要导师明确；代码是 MIT。
2. **探针标定常数**：`docs/data_quality.md` §4 的记录说明现有常数不适用于本批数据。数据公开后别人一定会问；
   要么拿到正确常数，要么在 `DATASHEET.md` 里明确写"标定常数未知，仅可做相对比较"。
3. **个人信息清理**：`20250609 LC` 的 `.ini`/`-st.txt` 里可能含操作员/设备信息，`.pvd` 头部含采集时间戳。
   建议公开前扫一遍（`scripts/09` 会列出所有非数值字段供人工确认）。

## 4. 别人拿到数据后 5 分钟能复现什么

```bash
git clone https://github.com/<user>/fluidized-bed-ml && cd fluidized-bed-ml
pip install -r requirements.txt
export FBFML_DATA_ROOT=/path/to/fbf-data
python scripts/03_wavelet_validation.py    # 54/54 完全吻合
python scripts/02_pressure_regime.py       # 0.55-0.65 m/s
python scripts/07_regime_classification.py # 82% vs 59% 基线
python scripts/08_verify_claims.py         # 20/20 自检通过
```

这就是"可复现研究"的完整闭环：**数据 + 代码 + 结论 + 自检**。

## 5. 不建议做的事

* 不要把 `.mat`/`.pvd` 硬塞进 Git 仓库（哪怕用 LFS）——仓库会变得极难 clone，而 Zenodo 本来就是为这个设计的。
* 不要在没有邮件确认的情况下把实验室数据标成你自己的版权。
* 不要删掉 `results/tables/` 里的中间结果：它们是"结论可追溯"的证据，比数据本身更常被审稿人要求。
