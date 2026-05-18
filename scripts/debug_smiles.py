import argparse
import os
from collections import Counter

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


# UFF 力场对很多金属/重元素支持不好；这里列出常见金属元素，
# 后面用来快速标记“可能不适合 RDKit + UFF 自动 3D 建模”的样本。
METALS = {
    "Li", "Na", "K", "Rb", "Cs", "Fr",
    "Be", "Mg", "Ca", "Sr", "Ba", "Ra",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Al", "Ga", "In", "Tl", "Sn", "Pb", "Bi",
}


def analyze_smiles(smiles, seed=42, max_attempts=50):
    """分析一条 SMILES 是否适合 RDKit 生成 3D 构象。

    返回一个字典，每个字段都会写入最终的 CSV，方便定位坏样本。
    """
    result = {
        "smiles": smiles,  # 原始 SMILES 字符串
        "parse_ok": False,  # RDKit 是否能解析这个 SMILES
        "canonical_smiles": "",  # RDKit 标准化后的 SMILES，便于去重或人工检查
        "num_atoms": 0,  # 原子总数
        "num_heavy_atoms": 0,  # 重原子数，即非氢原子数
        "num_fragments": 0,  # 分子片段数；大于 1 通常表示盐、混合物或多个分子
        "has_dot": "." in str(smiles),  # SMILES 中的 "." 表示多个不相连片段
        "has_metal": False,  # 是否包含常见金属/重元素
        "has_isotope": False,  # 是否包含同位素标记，如 [214Pb]
        "has_bracket_atom": "[" in str(smiles) and "]" in str(smiles),  # 是否包含括号原子写法
        "embed_ok": False,  # 是否成功生成初始 3D 坐标
        "uff_params_ok": False,  # UFF 是否有足够参数描述这个分子
        "uff_opt_ok": False,  # UFF 是否成功优化 3D 构象
        "reason": "",  # 最终结果：ok 或失败原因
    }

    # 第一步：把 SMILES 字符串解析成 RDKit 的分子对象。
    # 解析失败通常说明 SMILES 本身格式不合法。
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        result["reason"] = "invalid_smiles"
        return result

    # 解析成功后，记录一些基础结构信息。
    result["parse_ok"] = True
    result["canonical_smiles"] = Chem.MolToSmiles(mol)
    result["num_atoms"] = mol.GetNumAtoms()
    result["num_heavy_atoms"] = mol.GetNumHeavyAtoms()
    result["num_fragments"] = len(Chem.GetMolFrags(mol))

    # 遍历分子中的原子，识别金属/重元素和同位素。
    # 这些结构经常导致 UFFTYPER 的 Unrecognized atom type / charge state。
    atoms = list(mol.GetAtoms())
    result["has_metal"] = any(atom.GetSymbol() in METALS for atom in atoms)
    result["has_isotope"] = any(atom.GetIsotope() != 0 for atom in atoms)

    # 极端保护：如果 RDKit 解析出了空分子，后续 3D 建模没有意义。
    if result["num_atoms"] == 0:
        result["reason"] = "zero_atoms"
        return result

    try:
        # 3D 构象通常需要显式氢，所以先给分子补氢。
        mol_h = Chem.AddHs(mol)

        # EmbedMolecule 会根据键连接、环、手性等约束生成一个初始 3D 构象。
        # 返回值 0 表示成功，非 0 表示没有找到合适的 3D 构象。
        embed_code = AllChem.EmbedMolecule(
            mol_h,
            randomSeed=seed,
            maxAttempts=max_attempts,
        )

        if embed_code != 0:
            result["reason"] = "embed_failed"
            return result

        result["embed_ok"] = True

        # UFF 是一种通用分子力场，用来优化 3D 坐标。
        # 金属、重元素、特殊电荷状态经常没有 UFF 参数。
        result["uff_params_ok"] = AllChem.UFFHasAllMoleculeParams(mol_h)

        if not result["uff_params_ok"]:
            result["reason"] = "uff_missing_params"
            return result

        # 对初始 3D 构象做力场优化，让键长、键角、空间排布更合理。
        # RDKit 中 0 通常表示收敛，1 表示达到最大迭代但仍可作为可用构象看待。
        opt_code = AllChem.UFFOptimizeMolecule(mol_h)
        result["uff_opt_ok"] = opt_code in (0, 1)

        if not result["uff_opt_ok"]:
            result["reason"] = f"uff_opt_failed_code_{opt_code}"
        else:
            result["reason"] = "ok"

    except Exception as exc:
        # 捕获 RDKit 内部异常，避免单条坏样本导致整个脚本崩溃。
        result["reason"] = f"exception:{type(exc).__name__}:{exc}"

    return result


def analyze_file(csv_path, seed, max_attempts):
    """分析一个 CSV 文件中的全部 SMILES。"""
    # 读取 CSV。当前项目的数据列名应该包含 SMILES、Protein、Y。
    df = pd.read_csv(csv_path)

    if "SMILES" not in df.columns:
        raise ValueError(f"{csv_path} missing SMILES column")

    rows = []

    # fillna("") 可以避免空值导致 str/rdkit 处理时报错。
    for idx, smiles in enumerate(df["SMILES"].fillna("")):
        item = analyze_smiles(smiles, seed=seed, max_attempts=max_attempts)
        item["row_index"] = idx  # 原始 CSV 中的行号，方便回查
        item["source_file"] = csv_path  # 样本来源文件
        rows.append(item)

    # 每条 SMILES 的分析结果合并成一个 DataFrame，方便保存 CSV。
    return pd.DataFrame(rows)


def main():
    # 命令行参数：
    # --data-dir 指向包含 train.csv/val.csv/test.csv 的目录；
    # --out 指定调试结果输出目录。
    parser = argparse.ArgumentParser(description="Debug whether SMILES can build RDKit 3D conformers.")
    parser.add_argument("--data-dir", default="data/sample/random", help="directory containing train/val/test csv files")
    parser.add_argument("--out", default="outputs/debug_smiles", help="directory to save debug csv files")
    parser.add_argument("--seed", type=int, default=42, help="random seed for RDKit 3D embedding")
    parser.add_argument("--max-attempts", type=int, default=50, help="max RDKit embedding attempts per molecule")
    args = parser.parse_args()

    # 创建输出目录，exist_ok=True 表示目录已存在也不报错。
    os.makedirs(args.out, exist_ok=True)

    all_results = []

    # 依次检查训练集、验证集、测试集。
    # 如果某个文件不存在就跳过，所以也可以用 --data-dir data/sample 只测 test.csv。
    for name in ["train.csv", "val.csv", "test.csv"]:
        path = os.path.join(args.data_dir, name)
        if not os.path.exists(path):
            print(f"[skip] missing {path}")
            continue

        result = analyze_file(path, seed=args.seed, max_attempts=args.max_attempts)
        result["split"] = name.replace(".csv", "")
        all_results.append(result)

    # 如果三个文件都不存在，说明 data-dir 传错了。
    if not all_results:
        raise RuntimeError(f"No csv files found under {args.data_dir}")

    # 合并所有 split 的结果。
    result_df = pd.concat(all_results, ignore_index=True)

    # reason != ok 的样本都认为是 3D 建模风险样本。
    bad_df = result_df[result_df["reason"] != "ok"].copy()

    result_path = os.path.join(args.out, "smiles_debug_all.csv")
    bad_path = os.path.join(args.out, "smiles_debug_bad.csv")

    # smiles_debug_all.csv：所有样本的检查结果。
    # smiles_debug_bad.csv：只保存失败/风险样本，方便人工排查。
    result_df.to_csv(result_path, index=False)
    bad_df.to_csv(bad_path, index=False)

    # 控制台打印汇总信息，让你不打开 CSV 也能快速看结果。
    print("=== SMILES Debug Summary ===")
    print(f"data_dir: {args.data_dir}")
    print(f"total: {len(result_df)}")
    print(f"bad: {len(bad_df)}")
    print(f"all_result: {result_path}")
    print(f"bad_result: {bad_path}")
    print()

    print("reason counts:")
    for reason, count in Counter(result_df["reason"]).most_common():
        print(f"  {reason}: {count}")

    print()
    print("risk counts:")
    # 这些风险标记不一定代表必然失败，但出现越多，越说明数据不适合直接做 3D 建模。
    print(f"  multi_fragment: {int((result_df['num_fragments'] > 1).sum())}")
    print(f"  has_dot: {int(result_df['has_dot'].sum())}")
    print(f"  has_metal: {int(result_df['has_metal'].sum())}")
    print(f"  has_isotope: {int(result_df['has_isotope'].sum())}")
    print(f"  has_bracket_atom: {int(result_df['has_bracket_atom'].sum())}")

    if len(bad_df) > 0:
        print()
        print("first bad examples:")
        cols = ["split", "row_index", "reason", "smiles"]
        # 只打印前 20 条坏样本，完整列表看 smiles_debug_bad.csv。
        print(bad_df[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    # Python 脚本入口。直接运行 python scripts/debug_smiles.py 时会执行 main()。
    main()