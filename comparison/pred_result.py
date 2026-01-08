import argparse
import pickle
import numpy as np
import json
from pathlib import Path
from datetime import datetime

KABR_ACTIONS = ["Walk", "Graze", "Browse", "Head Up", "Auto-Groom", "Trot", "Run", "Occluded"]
PETACTION_ACTIONS = [
    "Running", "Walking", "Sniffing", "Standing(on all fours)", "Standing(bipedal)",
    "Sitting", "Lying", "Coughing", "Seizures", "Vomiting", "Movement Disorder"
]

def get_actions(dataset: str):
    ds = dataset.strip().lower()
    if ds == "kabr":
        return KABR_ACTIONS
    if ds == "petaction":
        return PETACTION_ACTIONS
    raise ValueError(f"Unsupported --dataset {dataset}. Use KABR or PetAction.")

def load_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def parse_index_from_key(k: str):
    # expect "test_123" / "train_5"
    try:
        return int(k.split("_")[-1])
    except Exception:
        return None

def infer_split_from_keys(keys):
    t = sum(1 for k in keys if str(k).startswith("test_"))
    tr = sum(1 for k in keys if str(k).startswith("train_"))
    if t >= tr and t > 0:
        return "test"
    if tr > t and tr > 0:
        return "train"
    return "test"

def build_ann_index(ann_list, fold: int, split: str):
    if split == "test":
        lst = [a for a in ann_list if int(a.get("fold", -999)) == int(fold)]
    elif split == "train":
        lst = [a for a in ann_list if int(a.get("fold", -999)) != int(fold)]
    else:
        raise ValueError("split must be 'test' or 'train'")
    # IMPORTANT: match your data-building order (sorted by sample_id)
    return sorted(lst, key=lambda a: int(a.get("sample_id", -1)))

def safe_label(actions, idx: int):
    if 0 <= idx < len(actions):
        return actions[idx]
    return f"unknown_{idx}"

def ensure_dict_scores(obj, name):
    if not isinstance(obj, dict):
        raise TypeError(f"{name} is not a dict. This script expects dict {{key: score_vector}}.")
    return obj

def sorted_keys(keys):
    keys = list(keys)
    keys_with_idx = [(k, parse_index_from_key(str(k))) for k in keys]
    if all(idx is not None for _, idx in keys_with_idx):
        return [k for k, _ in sorted(keys_with_idx, key=lambda x: x[1])]
    return sorted(keys, key=str)

def main(
    ann_json, fold, split, dataset, output_dir, print_first_n,
    pkl_joint, pkl_bone, pkl_joint_vel, pkl_bone_vel,
    w_joint, w_bone, w_joint_vel, w_bone_vel,
    fusion
):
    actions = get_actions(dataset)

    # Load annotation
    ann_list = json.loads(Path(ann_json).read_text())
    ann_sorted = build_ann_index(ann_list, fold=fold, split=split)

    # Load available score pkls
    streams = []  # (name, score_dict, weight)
    if pkl_joint:
        streams.append(("joint", ensure_dict_scores(load_pickle(pkl_joint), "joint pkl"), w_joint))
    if pkl_bone:
        streams.append(("bone", ensure_dict_scores(load_pickle(pkl_bone), "bone pkl"), w_bone))
    if pkl_joint_vel:
        streams.append(("joint_vel", ensure_dict_scores(load_pickle(pkl_joint_vel), "joint_vel pkl"), w_joint_vel))
    if pkl_bone_vel:
        streams.append(("bone_vel", ensure_dict_scores(load_pickle(pkl_bone_vel), "bone_vel pkl"), w_bone_vel))

    if len(streams) == 0:
        raise ValueError("You must provide at least one of --pkl_joint/--pkl_bone/--pkl_joint_vel/--pkl_bone_vel")

    # Decide split automatically only if user asked auto
    if split == "auto":
        # pick first stream to infer
        split = infer_split_from_keys(list(streams[0][1].keys()))
        ann_sorted = build_ann_index(ann_list, fold=fold, split=split)

    # Align keys across streams (intersection)
    key_sets = [set(d.keys()) for _, d, _ in streams]
    common_keys = set.intersection(*key_sets)

    if len(common_keys) == 0:
        raise ValueError("No common keys across provided pkls. Make sure they are from the same fold/split and same sample order.")

    keys_sorted = sorted_keys(common_keys)

    if len(keys_sorted) != len(ann_sorted):
        print(f"[WARN] common_keys={len(keys_sorted)} but annotation({split}, fold={fold})={len(ann_sorted)}.")
        print("       Will map by index up to min length. (Check fold/split & preprocessing order.)")

    n = min(len(keys_sorted), len(ann_sorted))

    # Fused prediction
    results = []
    for i in range(n):
        key = keys_sorted[i]

        fused = None
        used_streams = []
        for name, d, w in streams:
            sc = np.asarray(d[key], dtype=np.float32)
            if fused is None:
                fused = np.zeros_like(sc, dtype=np.float32)
            fused += float(w) * sc
            used_streams.append(name)

        if fusion == "avg":
            fused = fused / max(len(streams), 1)

        pred_id = int(fused.argmax())
        ann = ann_sorted[i]
        gt_id = int(ann.get("action_id", -1))

        item = {
            "score_key": str(key),
            "mapped_index": i,
            "sample_id": ann.get("sample_id"),
            "video_name": ann.get("video_name"),
            "source_video_id": ann.get("source_video_id"),
            "window_index": ann.get("window_index"),
            "fold": ann.get("fold"),
            "gt_id": gt_id,
            "gt_label": safe_label(actions, gt_id),
            "pred_id": pred_id,
            "pred_label": safe_label(actions, pred_id),
            "fusion": fusion,
            "streams": used_streams,
            "weights": {name: float(w) for name, _, w in streams},
        }
        results.append(item)

        if print_first_n > 0 and i < print_first_n:
            print(
                f"{item['score_key']:>8} | sample_id={item['sample_id']} | video={item['video_name']} | "
                f"win={item['window_index']} | gt={item['gt_label']} | pred={item['pred_label']} | streams={used_streams}"
            )

    # Write JSON
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    out_path = output_dir / f"preds_fused_top1_{split}_fold{fold}_{dataset}_{ts}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {len(results)} items to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--ann_json", required=True)
    ap.add_argument("--fold", default=0)
    ap.add_argument("--split", default="test", choices=["auto", "test", "train"])
    ap.add_argument("--dataset", default="PetAction", choices=["KABR", "PetAction"])
    ap.add_argument("--output_dir", default="output")
    ap.add_argument("--print_first_n", type=int, default=0)

    # score pkls (any subset)
    ap.add_argument("--pkl_joint", default=None)
    ap.add_argument("--pkl_joint_vel", default=None)
    ap.add_argument("--pkl_bone", default=None)
    ap.add_argument("--pkl_bone_vel", default=None)

    # weights (default 1.0)
    ap.add_argument("--w_joint", type=float, default=1.0)
    ap.add_argument("--w_bone", type=float, default=1.0)
    ap.add_argument("--w_joint_vel", type=float, default=1.0)
    ap.add_argument("--w_bone_vel", type=float, default=1.0)

    # fusion mode
    ap.add_argument("--fusion", default="sum", choices=["sum", "avg"], help="logits sum or average")

    args = ap.parse_args()

    main(
        ann_json=args.ann_json,
        fold=args.fold,
        split=args.split,
        dataset=args.dataset,
        output_dir=args.output_dir,
        print_first_n=args.print_first_n,
        pkl_joint=args.pkl_joint,
        pkl_bone=args.pkl_bone,
        pkl_joint_vel=args.pkl_joint_vel,
        pkl_bone_vel=args.pkl_bone_vel,
        w_joint=args.w_joint,
        w_bone=args.w_bone,
        w_joint_vel=args.w_joint_vel,
        w_bone_vel=args.w_bone_vel,
        fusion=args.fusion,
    )
