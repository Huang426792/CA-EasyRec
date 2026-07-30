# 上传到 GitHub

目标仓库：

```text
https://github.com/Huang426792/CA-EasyRec
```

## 方法一：使用 Git 命令（推荐）

解压 `CA-EasyRec-upload.zip`，进入解压后的目录，然后执行：

```bash
git init
git branch -M main
git add .
git commit -m "Initial release of CA-EasyRec"
git remote add origin https://github.com/Huang426792/CA-EasyRec.git
git push -u origin main
```

如果 GitHub 仓库已经存在 README，最后一条命令可能提示远端不是空仓库。
此时先执行：

```bash
git pull origin main --allow-unrelated-histories
```

解决可能出现的文件冲突、提交后，再执行：

```bash
git push -u origin main
```

由于建议创建的是空仓库，正常情况下不需要这一步。

GitHub 已不支持使用账户密码执行命令行推送。如果出现登录窗口，选择浏览器
登录；如果终端要求密码，应使用 Personal Access Token，而不是账户密码。

## 方法二：网页上传

1. 打开目标仓库。
2. 点击 `Add file` → `Upload files`。
3. 将解压后的文件和文件夹拖入网页。
4. 提交说明填写 `Initial release of CA-EasyRec`。
5. 点击 `Commit changes`。

项目包含多层目录，Git 命令比网页上传更稳定。

## 上传后检查

仓库首页应能看到：

```text
README.md
README_zh.md
src/ca_easyrec/
tests/
integration/
paper/
pyproject.toml
LICENSE
```

然后打开仓库的 `Actions` 页面，确认 `tests` 工作流运行成功。
