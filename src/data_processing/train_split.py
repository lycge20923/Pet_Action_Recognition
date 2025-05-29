import os
import json
import logging
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict, Counter

from sklearn.model_selection import StratifiedGroupKFold

from ..utils.cli_args import DataArguments
from ..utils.logging_utils import setup_logger


def generate_and_save_windows_once(annotations, window_size, num_samples, np_save_dir):
    """
    For each clip annotation, slice into windows, sample frames,
    save each window's features once, and return a flat list of metadata.
    """
    all_meta = []
    sample_id = 0
    for ann in tqdm(annotations, desc="Saving windows"):
        data = np.load(ann["feature_file_path"])
        kps = data["keypoints"]
        ofs = data["optical_flows"]
        n_windows = kps.shape[0] // window_size

        for w in range(n_windows):
            idxs = np.linspace(
                w * window_size,
                (w + 1) * window_size - 1,
                num_samples,
                dtype=int
            )
            kp_s = kps[idxs]
            of_s = ofs[idxs]

            feat_path = os.path.join(np_save_dir, f"{sample_id:06d}.npz")
            np.savez_compressed(
                feat_path,
                keypoints=kp_s,
                optical_flows=of_s
            )

            all_meta.append({
                "sample_id":        sample_id,
                "feature_file":     feat_path,
                "video_name":       ann["video_name"],
                "source_video_id":  ann["source_video_id"],
                "action_id":        ann["action_id"],
                "window_index":     w
            })
            sample_id += 1

        del data, kps, ofs
    return all_meta


def compute_basic_stats(annotations, logger):
    logger.info(f"Raw annotations: {len(annotations)} clip videos")
    cats = [a['action_id'] for a in annotations]
    logger.info(f"Action distribution across clips: {Counter(cats)}")
    srcs = [a['source_video_id'] for a in annotations]
    logger.info(f"Distinct source_video_id count: {len(set(srcs))}")


def main():
    logger = setup_logger(__file__, level=logging.INFO)
    args = DataArguments()
    base = os.path.splitext(args.annotation_file_name)[0]
    out_dir = os.path.join(args.data_dir, args.trainsplit_dir_name)
    np_save_dir = os.path.join(out_dir, "npz")
    os.makedirs(np_save_dir, exist_ok=True)

    # load and filter clip-level annotations
    ann_fp = os.path.join(
        args.data_dir,
        args.feature_extract_dir_name,
        args.annotation_file_name
    )
    with open(ann_fp) as f:
        raw_anns = json.load(f)
    compute_basic_stats(raw_anns, logger)
    raw_anns = [
        a for a in raw_anns if a.get('keypoint_detection_rate', 0) >= args.min_kp_rate
    ]
    logger.info(f"After filter: {len(raw_anns)} clip videos remain")

    # 1) window generation + save .npz once
    all_samples = generate_and_save_windows_once(
        raw_anns,
        window_size=args.window_size,
        num_samples=args.num_samples,
        np_save_dir=np_save_dir
    )
    # for temp test first
    # with open(os.path.join(out_dir, f"temp_windows_metadata.json"), 'r') as f:
    #     all_samples = json.load(f)
    
    # # for temp test second
    # with open(os.path.join(out_dir, f"temp_windows_metadata.json"), 'w') as f:
    #     json.dump(all_samples, f)  
     
    logger.info(f"Saved {len(all_samples)} window samples (.npz)")

    # 2) one-time StratifiedGroupKFold on source -> fold map
    sources = sorted({s['source_video_id'] for s in all_samples})
    # label each source by how many distinct actions it has
    src_labels = [
        len({s['action_id'] for s in all_samples if s['source_video_id'] == src})
        for src in sources
    ]
    sgkf = StratifiedGroupKFold(n_splits=args.fold_num, shuffle=True, random_state=42)
    source2fold = {}
    for fold, (_, val_idx) in enumerate(sgkf.split(sources, src_labels, groups=sources)):
        for idx in val_idx:
            source2fold[sources[idx]] = fold

    # 3) build per-(action, source) queues
    queues = defaultdict(lambda: defaultdict(list))
    for s in all_samples:
        queues[s['action_id']][s['source_video_id']].append(s)
    # sort each queue by sample_id
    for act, src_dict in queues.items():
        for src, lst in src_dict.items():
            random.shuffle(lst)

    # 4) round-robin sampling until difference > allow_diff
    allow_diff = getattr(args, 'fold_allow_diff', 0)
    counts = Counter({act: 0 for act in queues})
    final_samples = []
    round_k = 1
    while True:
        round_samples = []
        for act, src_dict in queues.items():
            for src, lst in src_dict.items():
                if len(lst) >= round_k:
                    samp = lst[round_k-1].copy()
                    samp['fold'] = source2fold[src]
                    round_samples.append(samp)
        if not round_samples:
            break
        # compute prospective counts
        temp_counts = counts.copy()
        for s in round_samples:
            temp_counts[s['action_id']] += 1
        # stop if difference exceeds threshold
        if max(temp_counts.values()) - min(temp_counts.values()) > allow_diff:
            break
        # accept this round
        final_samples.extend(round_samples)
        counts = temp_counts
        round_k += 1

    # log per-fold class distribution
    for fold in range(args.fold_num):
        cnt = Counter(s['action_id'] for s in final_samples if s['fold'] == fold)
        sorted_cnt = {action: cnt[action] for action in sorted(cnt)}
        logger.info(f"[Fold {fold}] samples per action: {sorted_cnt}")


    # save final metadata
    meta_fp = os.path.join(out_dir, f"{base}_windows_metadata.json")
    with open(meta_fp, 'w') as f:
        json.dump(final_samples, f, indent=2)
    logger.info(
        f"Generated {len(final_samples)} round-robin balanced samples (allow_diff={allow_diff})"
    )

if __name__ == '__main__':
    main()
