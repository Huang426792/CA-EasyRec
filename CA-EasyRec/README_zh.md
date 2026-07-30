# CA-EasyRec 中文说明

CA-EasyRec 是对 EasyRec 的一个小改进：利用冻结的 LightGCN 教师模型识别
可能的假负样本，降低这些样本在对比学习分母中的权重，避免文本编码器把潜在
感兴趣物品过度推远。

[返回英文主页](README.md) ·
[官方 EasyRec 接入说明](integration/EASYREC_INTEGRATION.md) ·
[GitHub 上传步骤](UPLOAD_TO_GITHUB_zh.md)

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

## 代码归属

本仓库只包含本项目原创实现，没有复制完整 EasyRec 源码。EasyRec 模型、
数据、画像或实验协议仍应引用其 EMNLP 2025 论文和官方仓库。
