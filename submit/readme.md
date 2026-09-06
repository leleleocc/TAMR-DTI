# TAMR-DTI 投稿文件说明

本目录 `submit` 保存向期刊投稿系统上传的最终文件。论文的完整 LaTeX 工作目录是：

`D:\Code\TAMR-DTI\TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction`

## 两个目录的关系

长名称目录是论文的源文件和编译工作目录，包含主论文、章节文件、参考文献、样式文件、图片、SI 的 LaTeX 源文件以及编译生成的 PDF 和临时文件。

`submit\manuscript.zip` 是从上述源目录整理后生成的投稿源文件包。它保留了能够重新编译论文所需的 `.tex`、`.bib`、`.bst`、`.sty`、章节文件和图像文件，同时包含 SI 的 LaTeX 源文件；编译生成的 PDF、`.aux`、`.log`、`.out`、`.toc`、`.synctex` 等临时或重复文件没有放入压缩包。

因此，`manuscript.zip` 与长名称目录的关系是：

```text
论文源目录（完整工作目录）
        |
        +-- 筛选可提交的 LaTeX 源文件和图像
        |
        +-- submit\manuscript.zip
```

`manuscript.zip` 不是对长名称目录的原样整体压缩，而是该目录的干净提交版本。长名称目录仍然是后续修改和重新编译的主目录；修改源文件后，如果需要更新投稿包，应重新从源目录生成 `manuscript.zip`。

## 当前投稿文件

| 文件 | 投稿系统中的位置 | 说明 |
|---|---|---|
| `manuscript.zip` | **Manuscript File** | 主论文 LaTeX 源文件包，包含编译所需资源和 SI 源码 |
| `Cover Letter.docx` | **Cover Letter** | 投稿信 |
| `TAMR-DTI_Supporting_Information.pdf` | **Other Files** | 最终 SI PDF，包含旧 SI 内容、公式解释表、独立表格和高清图 |

## 上传顺序

1. 在 **Manuscript File** 中上传 `manuscript.zip`。
2. 在 **Cover Letter** 中上传 `Cover Letter.docx`。
3. 在 **Other Files** 中上传 `TAMR-DTI_Supporting_Information.pdf`，并按系统要求将其 designation 设为 Supporting Information（或最接近的 SI 选项）。

原来的 `TAMR-DTI_SI_submission_20260906b.zip` 不再需要上传，因为 SI 的最终 PDF 已经单独列出，SI 的源文件也已经包含在 `manuscript.zip` 中。

## 备注

- 主论文中的 `Supporting Information` 简短说明用于提示读者 SI 的内容；完整 SI 作为独立 PDF 提交。
- `manuscript.zip` 不需要再额外上传主论文 PDF；投稿系统会根据压缩包中的源文件生成或检查稿件。
- 若只需要阅读或检查 SI，直接打开 `TAMR-DTI_Supporting_Information.pdf` 即可。
