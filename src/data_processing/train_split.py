import os
import json
import logging
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
import math

from sklearn.model_selection import StratifiedGroupKFold

from ..utils.cli_args import DataArguments
from ..utils.logging_utils import setup_logger
from ..utils.common import set_comparison_config, set_comparison_config_args

def load_video2fold_map(ref_meta_path, logger):
    """
    This is for the ablation study of preprocessing 
    """
    with open(ref_meta_path, "r") as f:
        ref_meta = json.load(f)
    video2fold = {}
    missing_fold = 0
    for item in ref_meta:
        vname = item.get("video_name")
        fold = item.get("fold")
        if vname not in video2fold:
            video2fold[vname] = int(fold)
    
    logger.info(f"[fold_ref] Loaded {len(video2fold)} video_name->fold mappings ")
    return video2fold

def build_source_action_vectors(all_samples):
    """
    sources:  sorted source list
    actions:  sorted action list
    src2vec:  dict[src] -> np.array([A]), window counts of each action
    total:    total action vectors
    """
    sources = sorted({s['source_video_id'] for s in all_samples})
    actions = sorted({s['action_id'] for s in all_samples})
    A = len(actions)
    act2idx = {a:i for i,a in enumerate(actions)}

    per_sa = defaultdict(int)
    for s in all_samples:
        per_sa[(s['source_video_id'], s['action_id'])] += 1

    src2vec = {}
    for src in sources:
        v = np.zeros(A, dtype=int)
        for a in actions:
            v[act2idx[a]] = per_sa[(src, a)]
        src2vec[src] = v

    total = np.zeros(A, dtype=int)
    for v in src2vec.values():
        total += v
    return sources, actions, src2vec, total

def assign_sources_to_folds_balanced(sources, src2vec, total, K=5, seed=42, coverage_bonus=1.0):
    """
    Group-aware multi-class stratification with a global L1-deviation objective, 
    plus a coverage bonus for assignments that fill classes missing in a fold.
    - Objective: minimize  sum_j ||fold_j - target||_1
    - Coverage bonus:
    if assigning a source to a fold introduces any class that currently has zero count in that fold, 
    decrease the score (i.e., make that assignment more favorable).
    - Guarantee: ensure every fold receives at least one source; 
    if any fold is left empty, perform a one-time rebalancing/shift from the most overloaded fold.
    """
    rng = np.random.default_rng(seed)
    target = total.astype(float) / float(K)
    folds = [np.zeros_like(target, dtype=float) for _ in range(K)]
    assign = {}

    # First deal with "big" source
    src_order = sorted(sources, key=lambda s: src2vec[s].sum(), reverse=True)

    for s in src_order:
        v = src2vec[s].astype(float)
        best_k, best_score = None, None
        for k in range(K):
            old = folds[k].copy()
            before = np.abs(old - target).sum()
            after  = np.abs((old + v) - target).sum()
            delta  = after - before  # put s into k, enlarge L1 bigger(but make it smaller is better)

            # coverage
            new_covered = int(((old == 0) & (v > 0)).sum())
            score = delta - coverage_bonus * new_covered

            # break even
            if (best_score is None) or (score < best_score) or (np.isclose(score, best_score) and rng.random() < 0.5):
                best_score, best_k = score, k

        folds[best_k] += v
        assign[s] = best_k

    # Ensure every fold is used: 
    # if any fold is empty, move the smallest source from the most overloaded fold to that fold.
    used = set(assign.values())
    if len(used) < K:
        empties = [k for k in range(K) if k not in used]
        for k in empties:
            # Select a donor: the fold that currently has the largest L1 deviation.
            donor = max(range(K), key=lambda j: np.abs(folds[j] - target).sum())
            cand = [s for s, jj in assign.items() if jj == donor]
            # Move the source with the smallest total count to minimize disruption.
            s_move = min(cand, key=lambda s: src2vec[s].sum())
            folds[donor] -= src2vec[s_move]
            folds[k]     += src2vec[s_move]
            assign[s_move] = k

    return assign

def generate_and_save_windows_once(annotations, window_size, num_samples, np_save_dir, for_comparison, add_rgb=False, segment_dir=None):
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
        if for_comparison:
            n_windows = math.ceil(kps.shape[0] / window_size)
        else:
            n_windows = kps.shape[0] // window_size
        
        kp_buf = np.empty((num_samples, *kps.shape[1:]), dtype=kps.dtype)
        of_buf = np.empty((num_samples, *ofs.shape[1:]), dtype=ofs.dtype)
        
        # add rgb for ablation study
        ref_rgb = add_rgb and segment_dir is not None
        if ref_rgb:
            rgbs = data["rgbs"]
            rgb_buf = np.empty((num_samples, *rgbs.shape[1:]), dtype=rgbs.dtype)  # [num_samples, H, W, 3]
            

        for w in range(n_windows):
            if for_comparison:
                idxs = np.linspace(
                    w * window_size,
                    (w + 1) * window_size - 1,
                    num_samples,
                    dtype=int
                )
                idxs = np.clip(idxs, 0, kps.shape[0] - 1)
            else:
                idxs = np.linspace(
                    w * window_size,
                    (w + 1) * window_size - 1,
                    num_samples,
                    dtype=int
                )
            
            np.take(kps, idxs, axis=0, out=kp_buf)
            np.take(ofs, idxs, axis=0, out=of_buf)
            
            kp_feat_path = os.path.join(np_save_dir, f"{sample_id:06d}_kp.npy")
            of_feat_path = os.path.join(np_save_dir, f"{sample_id:06d}_of.npy")
            
            np.save(kp_feat_path, kp_buf)
            np.save(of_feat_path, of_buf)
            
            store_dict = {
                "sample_id":        sample_id,
                "kp_feature_file":     kp_feat_path,
                "of_feature_file": of_feat_path, 
                "video_name":       ann["video_name"],
                "source_video_id":  ann["source_video_id"],
                "action_id":        ann["action_id"],
                "window_index":     w
            }
            if ref_rgb:
                np.take(rgbs, idxs, axis=0, out=rgb_buf)
                rgb_feat_path = os.path.join(np_save_dir, f"{sample_id:06d}_rgb.npy")
                np.save(rgb_feat_path, rgb_buf)
                store_dict["rgb_feature_file"] = rgb_feat_path
                
            if "split" in ann.keys(): # for comparison other 
                store_dict["fold"] = 0 if ann["split"] == "val" else 1
                
            all_meta.append(store_dict)
            
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
    data_args = DataArguments()
    
    comparison_args = set_comparison_config_args()
    if comparison_args.for_comparison:
        data_args = set_comparison_config(data_args, comparison_args.dataset_name)
    
    base = os.path.splitext(data_args.annotation_file_name)[0]
    
    
    out_dir = os.path.join(data_args.data_dir, data_args.trainsplit_dir_name)
    np_save_dir = os.path.join(out_dir, "npy")
    os.makedirs(np_save_dir, exist_ok=True)

    # load and filter clip-level annotations
    ann_fp = os.path.join(
        data_args.data_dir,
        data_args.feature_extract_dir_name,
        data_args.annotation_file_name
    )
    with open(ann_fp) as f:
        raw_anns = json.load(f)
    if not comparison_args.for_comparison:
        compute_basic_stats(raw_anns, logger)
        
    if data_args.feature_extract_dir_name == "feature_extracted" and not comparison_args.for_comparison: # only for the complete experiment, not for ablation study
        raw_anns = [
            a for a in raw_anns if a.get('keypoint_detection_rate', 0) >= data_args.min_kp_rate
        ]
    logger.info(f"After filter: {len(raw_anns)} clip videos remain")

    # 1) window generation + save .npz once
    all_samples = generate_and_save_windows_once(
        raw_anns,
        window_size=data_args.window_size,
        num_samples=data_args.num_samples,
        np_save_dir=np_save_dir,
        for_comparison=comparison_args.for_comparison,
        add_rgb=data_args.rgb_include,
        segment_dir=os.path.join(data_args.data_dir, data_args.seg_dir_name)
    )
    # for temp test
    # with open(os.path.join(out_dir, f"temp_windows_metadata.json"), 'r') as f:
    #     all_samples = json.load(f)
    
    # with open(os.path.join(out_dir, f"temp_windows_metadata.json"), 'w') as f:
    #     json.dump(all_samples, f)  
     
    logger.info(f"Saved {len(all_samples)} window samples (.npz)")
    if not comparison_args.for_comparison:
        final_samples = []
        # 2) Group-aware multi-class stratification on sources -> fold map
        if data_args.feature_extract_dir_name != "feature_extracted": # for ablation study of stabilization & cropping 
            ref_metadata_path = "data/main/train_split/annotation_windows_metadata.json"
            if not os.path.exists(ref_metadata_path):
                raise ValueError("For fair comparison, the dataset distribution of 5 folds should be the same among different experiments")
            video2fold = load_video2fold_map(
                ref_metadata_path,
                logger
            )
            missing = 0
            for s in all_samples:
                vname = s["video_name"]
                if vname not in video2fold:
                    missing += 1
                    logger.warning(
                        f"[fold_ref] video_name '{vname}' not found in reference folds. "
                        f"Sample_id={s['sample_id']} will be skipped."
                    )
                    continue

                item = s.copy()
                item["fold"] = video2fold[vname]
                final_samples.append(item)

            logger.info(
                f"[fold_ref] Assigned folds from reference json for "
                f"{len(final_samples)} samples, skipped {missing} samples without mapping."
            )

            # logging per-fold 統計
            fold_srcs = defaultdict(set)
            for it in final_samples:
                fold_srcs[it["fold"]].add(it["source_video_id"])

            for fold in range(data_args.num_folds):
                cnt = Counter(s["action_id"] for s in final_samples if s["fold"] == fold)
                sorted_cnt = {a: cnt.get(a, 0) for a in sorted({s["action_id"] for s in final_samples})}
                n_src = len(fold_srcs.get(fold, []))
                n_samp = sum(sorted_cnt.values())
                logger.info(f"[Fold {fold}] #sources={n_src}, #samples={n_samp}, per-action={sorted_cnt}")
        else:
            sources, actions, src2vec, total = build_source_action_vectors(all_samples)
            source2fold = assign_sources_to_folds_balanced(
                sources, src2vec, total, K=data_args.num_folds, seed=42, coverage_bonus=1.0
            )
            
            for s in all_samples:
                item = s.copy()
                item['fold'] = source2fold[s['source_video_id']]
                final_samples.append(item)
        
            # for logging
            fold_srcs = defaultdict(list)
            for src, f in source2fold.items():
                fold_srcs[f].append(src)
            
            for fold in range(data_args.num_folds):
                cnt = Counter(s['action_id'] for s in final_samples if s['fold'] == fold)
                sorted_cnt = {a: cnt.get(a,0) for a in sorted({s['action_id'] for s in all_samples})}
                n_src = len(fold_srcs.get(fold, []))
                n_samp = sum(sorted_cnt.values())
                logger.info(f"[Fold {fold}] #sources={n_src}, #samples={n_samp}, per-action={sorted_cnt}")
    else:
        final_samples = all_samples
    # save final metadata
    meta_fp = os.path.join(out_dir, f"{base}_windows_metadata.json")
    with open(meta_fp, 'w') as f:
        json.dump(final_samples, f, indent=2)
    logger.info(
        f"Generated {len(final_samples)} samples."
    )

if __name__ == '__main__':
    main()
