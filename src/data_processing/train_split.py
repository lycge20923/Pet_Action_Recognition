import os
import json
import random
import logging
from collections import defaultdict, Counter

import numpy as np
from sklearn.model_selection import StratifiedKFold

from ..utils.cli_args import DataArguments
from ..utils.logging_utils import setup_logger

def normalize_keypoints(keypoints, bboxes, num_nodes):
    """
    Normalize each keypoint in a frame relative to its bounding box.
    If either keypoints or bboxes is empty, return a single [0,0,0].
    """
    if not keypoints or not bboxes or len(bboxes) != 4:
        return [[0.0, 0.0, 0.0] for _ in range(num_nodes)]
    x1, y1, x2, y2 = bboxes
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return [[0.0, 0.0, 0.0] for _ in range(num_nodes)]
    normalized = []
    for x, y, c in keypoints:
        nx = (x - x1) / width
        ny = (y - y1) / height
        normalized.append([nx, ny, c])
    return normalized

def sample_frames_from_window(window_data, num_samples):
    """Evenly pick `num_samples` frames from `window_data`."""
    idx = np.linspace(0, len(window_data) - 1, num_samples, dtype=int)
    return [window_data[i] for i in idx]

def generate_window_samples(annotations, window_size, num_samples, num_nodes):
    """For each ann, split its frames into windows, sample frames, then normalize keypoints."""
    by_cat = defaultdict(list)
    for ann in annotations:
        frames = ann["data"]
        n_windows = len(frames) // window_size
        for w in range(n_windows):
            window = frames[w*window_size : (w+1)*window_size]
            sampled = sample_frames_from_window(window, num_samples)
            # normalize every sampled frame's keypoints
            normalized_data = []
            for frame in sampled:
                kps = frame.get("keypoints", [])
                bbox = frame.get("bboxes", [])
                normalized_data.append(normalize_keypoints(kps, bbox, num_nodes))
            sample = {
                "video_name":      ann["video_name"],
                "source_video_id": ann["source_video_id"],
                "action_id":     ann["action_id"],
                "window_index":    w,
                "data":            normalized_data,
                "sampled_frames":  sampled
            }
            by_cat[ann["action_id"]].append(sample)
    return by_cat

def balanced_category_sampling(samples_by_cat):
    """Downsample each category to the smallest size, preferring one per source."""
    min_n = min(len(lst) for lst in samples_by_cat.values())
    balanced = []
    for samples in samples_by_cat.values():
        grouped = defaultdict(list)
        for s in samples:
            grouped[s["source_video_id"]].append(s)
        pick = []
        for src in random.sample(list(grouped), len(grouped)):
            if len(pick) >= min_n:
                break
            pick.append(grouped[src][0])
        if len(pick) < min_n:
            rest = [s for s in samples if s not in pick]
            pick += random.sample(rest, min_n - len(pick))
        balanced.extend(pick)
    return balanced

def compute_basic_stats(annotations, logger):
    """Log counts, distributions of categories, sources, rates, frames."""
    logger.info(f"Total samples: {len(annotations)}")
    cats = [a["action_id"] for a in annotations]
    logger.info(f"Categories: {Counter(cats)}")
    srcs = [a["source_video_id"] for a in annotations]
    logger.info(f"Video sources: {len(set(srcs))}, top10: {Counter(srcs).most_common(10)}")
    rates = np.array([a.get("keypoint_detection_rate", 0) for a in annotations])
    logger.info(f"KPR min/max/mean/med: {rates.min():.3f}/{rates.max():.3f}/{rates.mean():.3f}/{np.median(rates):.3f}")
    lens = np.array([len(a["data"]) for a in annotations])
    logger.info(f"Frames per sample min/max/mean/med: {lens.min()}/{lens.max()}/{lens.mean():.1f}/{np.median(lens):.1f}")

def main():
    logger = setup_logger(__file__, level=logging.INFO)
    args   = DataArguments()
    base   = os.path.splitext(args.annotation_file_name)[0]
    out_dir = os.path.join(args.data_dir, args.trainsplit_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    # load + stats
    fp = os.path.join(args.data_dir, args.feature_extract_dir_name, args.annotation_file_name)
    with open(fp) as f:
        anns = json.load(f)
    logger.info(f"Loaded {len(anns)} raw annotations")
    compute_basic_stats(anns, logger)

    # filter by rate
    anns = [a for a in anns if a.get("keypoint_detection_rate", 0) >= args.min_kp_rate]
    logger.info(f"After filter: {len(anns)} samples")

    # stratify by source -> count of unique categories
    src2cats = defaultdict(set)
    for a in anns:
        src2cats[a["source_video_id"]].add(a["action_id"])
    sources = list(src2cats)
    labels  = [len(src2cats[s]) for s in sources]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, vl_idx) in enumerate(skf.split(sources, labels)):
        tr_src = {sources[i] for i in tr_idx}
        vl_src = {sources[i] for i in vl_idx}
        tr_anns = [a for a in anns if a["source_video_id"] in tr_src]
        vl_anns = [a for a in anns if a["source_video_id"] in vl_src]
        logger.info(f"[Fold{fold}] train_samples={len(tr_anns)}, val_samples={len(vl_anns)}")

        # train: window + balance
        by_cat = generate_window_samples(tr_anns, window_size=args.window_size, num_samples=args.num_samples, num_nodes=args.num_nodes)
        raw_counts = {c: len(l) for c, l in by_cat.items()}
        logger.info(f"[Fold{fold}] raw windows per cat: {raw_counts}")
        balanced = balanced_category_sampling(by_cat)
        random.shuffle(balanced)
        logger.info(f"[Fold{fold}] balanced windows total={len(balanced)}")

        with open(os.path.join(out_dir, f"{base}_fold{fold}_train_windows.json"), "w") as f:
            json.dump(balanced, f, indent=4)

        # val: window only
        by_cat_v = generate_window_samples(vl_anns, window_size=args.window_size, num_samples=args.num_samples, num_nodes=args.num_nodes)
        flat_v = [s for lst in by_cat_v.values() for s in lst]
        random.shuffle(flat_v)
        logger.info(f"[Fold{fold}] val windows total={len(flat_v)}")

        with open(os.path.join(out_dir, f"{base}_fold{fold}_val_windows.json"), "w") as f:
            json.dump(flat_v, f, indent=4)

if __name__ == "__main__":
    main()
