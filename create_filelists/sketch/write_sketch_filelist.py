#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from collections import defaultdict

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


def infer_class_name_from_relpath(rel_path: str):
    """
    DomainNet 通常是：sketch/<class_name>/<img>.png
    返回 class_name；如果结构不符合，返回 None
    """
    rel_norm = rel_path.replace("\\", "/").strip()
    toks = rel_norm.split("/")
    if len(toks) >= 2 and toks[0] == "sketch":
        return toks[1]
    return None


def build_split_meta(split_labels, label_to_paths, label_to_name):
    """
    split_labels: list[int]  该 split 的绝对 label 列表（按升序）
    label_to_paths: dict[int, list[str]]
    label_to_name: dict[int, str]
    返回 meta dict: {label_names, image_names, image_labels}
    """
    image_names = []
    image_labels = []
    label_names = []

    for lab in split_labels:
        cname = label_to_name.get(lab, f"class_{lab}")
        label_names.append(cname)
        for p in label_to_paths[lab]:
            image_names.append(p)
            image_labels.append(lab)  # 保持绝对 label，不重排

    return {
        "label_names": label_names,
        "image_names": image_names,
        "image_labels": image_labels,
    }


def save_classes_txt(path, split_labels, label_to_name):
    """
    输出格式：<abs_label>\t<class_name>
    """
    with open(path, "w", encoding="utf-8") as f:
        for lab in split_labels:
            cname = label_to_name.get(lab, f"class_{lab}")
            f.write(f"{lab}\t{cname}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt", type=str,
                        default="/data/pikey/dataset/FewshotData/DomainNet/sketch/sketch_test.txt",
                        help="Input txt metadata file (each line: <rel_path> <label_id> ...)")
    parser.add_argument("--prefix", type=str,
                        default="/data/pikey/dataset/FewshotData/DomainNet/",
                        help="Absolute path prefix to be prepended")

    # 输出目录：会在该目录下生成 base.json/val.json/novel.json 以及对应 classes.txt
    parser.add_argument("--out_dir", type=str,
                        default="/data/pikey/dataset/FewshotData/DomainNet/sketch/",
                        help="Output directory")

    parser.add_argument("--stride", type=int, default=4,
                        help="Grouping stride (default 4): [0,1]->base, [2]->val, [3]->novel")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 读取并按原始绝对 label_id 聚合
    label_to_paths = defaultdict(list)
    label_to_name = {}

    with open(args.txt, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue

            rel = parts[0].strip().rstrip(".")  # 防御性处理：偶尔末尾多一个 '.'
            lab_str = parts[1].strip()

            try:
                lab = int(lab_str)
            except ValueError:
                continue

            if not rel.lower().endswith(EXTS):
                continue

            abs_path = rel if rel.startswith(args.prefix) else os.path.join(args.prefix, rel)
            label_to_paths[lab].append(abs_path)

            cname = infer_class_name_from_relpath(rel)
            if cname is not None:
                label_to_name.setdefault(lab, cname)

    all_labels = sorted(label_to_paths.keys())
    num_classes = len(all_labels)
    if num_classes == 0:
        raise RuntimeError("No valid samples/classes found in txt.")

    # 2) 按 4 个一组划分：前2 base，第3 val，第4 novel
    base_labels, val_labels, novel_labels = [], [], []

    for idx, lab in enumerate(all_labels):
        pos = idx % args.stride  # 0,1,2,3,...

        # 正常组内规则：
        if pos in (0, 1):
            base_labels.append(lab)
        elif pos == 2:
            val_labels.append(lab)
        else:  # pos == 3
            novel_labels.append(lab)

    # 3) 处理“最后不满4个”的要求：
    # 由于上面是按 idx%4 切分，其实天然满足“先base再val再novel”。
    # 例如剩1个 -> idx%4==0 -> base
    # 剩2个 -> 0,1 -> base base
    # 剩3个 -> 0,1,2 -> base base val
    # novel 只有当凑到第4个（pos==3）才会有
    # ——这正好符合你的规则，因此无需额外处理。

    # 4) 生成并保存 json
    base_meta = build_split_meta(base_labels, label_to_paths, label_to_name)
    val_meta = build_split_meta(val_labels, label_to_paths, label_to_name)
    novel_meta = build_split_meta(novel_labels, label_to_paths, label_to_name)

    base_json = os.path.join(args.out_dir, "base.json")
    val_json = os.path.join(args.out_dir, "val.json")
    novel_json = os.path.join(args.out_dir, "novel.json")

    with open(base_json, "w", encoding="utf-8") as f:
        json.dump(base_meta, f)
    with open(val_json, "w", encoding="utf-8") as f:
        json.dump(val_meta, f)
    with open(novel_json, "w", encoding="utf-8") as f:
        json.dump(novel_meta, f)

    # 5) 保存 classes.txt（用于保证 train/test 类严格不同）
    base_cls = os.path.join(args.out_dir, "base_classes.txt")
    val_cls = os.path.join(args.out_dir, "val_classes.txt")
    novel_cls = os.path.join(args.out_dir, "novel_classes.txt")

    save_classes_txt(base_cls, base_labels, label_to_name)
    save_classes_txt(val_cls, val_labels, label_to_name)
    save_classes_txt(novel_cls, novel_labels, label_to_name)

    # 6) 打印统计
    def _stat(name, labels, meta):
        print(f"[{name}] #classes={len(labels)}, #images={len(meta['image_names'])}")
        if labels:
            print(f"    label_range=({labels[0]}..{labels[-1]}), head={labels[:8]}, tail={labels[-8:]}")

    print(f"[OK] Parsed total classes: {num_classes}")
    _stat("base", base_labels, base_meta)
    _stat("val", val_labels, val_meta)
    _stat("novel", novel_labels, novel_meta)

    print("[OK] Saved:")
    print(f"  {base_json}")
    print(f"  {val_json}")
    print(f"  {novel_json}")
    print("[OK] Saved class lists:")
    print(f"  {base_cls}")
    print(f"  {val_cls}")
    print(f"  {novel_cls}")


if __name__ == "__main__":
    main()
