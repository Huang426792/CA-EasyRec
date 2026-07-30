# CA-EasyRec 中文说明

CA-EasyRec 是对 EasyRec 的一个小改进：利用冻结的 LightGCN 教师模型识别
可能的假负样本，降低这些样本在对比学习分母中的权重，避免文本编码器把潜在
感兴趣物品过度推远。

[返回英文主页](README.md) ·
[官方 EasyRec 接入说明](integration/EASYREC_INTEGRATION.md) ·
[GitHub 上传步骤](UPLOAD_TO_GITHUB_zh.md) ·
[ACL 论文](paper/Di_Liu_NLP_Assignment.pdf)

> 仓库里的指标只是用于证明程序可以完整运行的 toy smoke test，不是
> Sports、Steam 或 Yelp 的论文实验结果。

## 核心公式

对用户 `u` 的同域负样本 `j`，先使用冻结的 LightGCN 计算协同得分
`q(u,j)`，再进行批内标准化：

```text
confidence = sigmoid((q - mean) / (std + delta))
weight = epsilon + (1 - epsilon) * (1 - confidence) ^ gamma
```

教师认为越可能相关的负样本，`weight` 越小。正样本、跨域负样本以及同域
候选少于两个的情况全部使用权重 1。教师得分不参与反向传播，测试目标域时
也不需要 LightGCN。

## 快速运行

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . --no-deps
python -m pip install numpy scipy
python -m ca_easyrec.demo --output artifacts/demo --teacher-epochs 5 --text-epochs 5
```

Linux：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . --no-deps
python -m pip install numpy scipy
python -m ca_easyrec.demo --output artifacts/demo --teacher-epochs 5 --text-epochs 5
```

运行后生成：

```text
artifacts/demo/teacher.pt
artifacts/demo/text_model.pt
artifacts/demo/metrics.json
```

## 运行测试

```bash
python -m unittest discover -s tests -v
```

## 在官方 EasyRec 数据上训练教师

先按照 [EasyRec 官方仓库](https://github.com/HKUDS/EasyRec)下载数据，再执行：

```bash
python -m ca_easyrec.official \
  --data-root /path/to/EasyRec/data \
  --domains arts games movies home electronics tools \
  --output artifacts/easyrec_teachers.pt \
  --embedding-dim 64 \
  --num-layers 2 \
  --epochs 100 \
  --device cuda:0
```

之后按照
[integration/EASYREC_INTEGRATION.md](integration/EASYREC_INTEGRATION.md)
修改官方数据读取器和 `model.py` 的损失计算部分。

## 论文正式实验

论文结果需要满足以下条件后才能填写：

1. 使用完全相同的用户/物品画像、batch 和数据划分复现 EasyRec；
2. EasyRec 与 CA-EasyRec 分别运行 5 个随机种子；
3. 在 Sports、Steam、Yelp 上进行全排序评估；
4. 报告 Recall@10/20、NDCG@10/20 的均值和标准差；
5. 完成假负样本审计、消融实验和 `epsilon/gamma` 敏感性分析。

在上述实验没有真正运行前，不应把 toy 指标填入 ACL 论文。

## 代码归属

本仓库只包含本项目原创实现，没有复制完整 EasyRec 源码。EasyRec 模型、
数据、画像或实验协议仍应引用其 EMNLP 2025 论文和官方仓库。
