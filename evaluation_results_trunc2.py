#!/usr/bin/env python3
import os
import json
import numpy as np
import argparse
import ast
import pandas as pd

def load_data(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def scale_data(data, scale=25):
    """Scale a list of 2D points by the given factor."""
    return [[p[0] * scale, p[1] * scale] for p in data]

def compute_centerline_and_halfwidths(lanes):
    """
    Given lanes (assumed that lanes[0] and lanes[1] are the road boundaries),
    compute a centerline and half-widths.
    The centerline is computed as the midpoint of corresponding points on lane0 and lane1.
    The half-width is the distance from the midpoint to lane0.
    """
    if lanes is None or len(lanes) < 2:
        return None, None
    lane0 = lanes[0]
    lane1 = lanes[1]
    n = min(len(lane0), len(lane1))
    centerline = []
    half_widths = []
    for i in range(n):
        cx = (lane0[i][0] + lane1[i][0]) / 2.0
        cy = (lane0[i][1] + lane1[i][1]) / 2.0
        centerline.append((cx, cy))
        hw = np.hypot(lane0[i][0] - cx, lane0[i][1] - cy)
        half_widths.append(hw)
    return centerline, half_widths

def compute_computed_cte(x, y, centerline, half_widths):
    """
    Compute the cross-track error (CTE) for a given (x,y) pose.
    For each point on the centerline, the Euclidean distance is computed and the minimum is selected.
    The distance is scaled so that a distance equal to the half-width corresponds to 1.5 CTE-units.
    """
    if centerline is None:
        return 0
    distances = [np.hypot(x - cx, y - cy) for cx, cy in centerline]
    idx = np.argmin(distances)
    closest_distance = distances[idx]
    hw = half_widths[idx]
    return (closest_distance / hw) * 1.5 if hw > 0 else 0

def analyze_run(poses_data, moving_data, survival_threshold, safe_thresholds):
    """
    Process a single run and compute:
      - SurvivedDeg: The cumulative turning (in degrees) from the beginning of the run until the computed CTE
                    first exceeds the survival threshold.
      - TotalTurn: The cumulative turning (in degrees) over the entire run (only counting "going" segments).
      - For each safe threshold, the percentage of turning (up to the survival threshold) where the computed CTE
        was at or below that safe threshold.
      - MeanSpeed: The mean speed computed from consecutive "going" poses.
      - MaxCTE: The maximum computed CTE observed during the run.
    
    The poses JSON must contain keys "pose", "time", and "lanes".
    The moving JSON must contain keys "going" and "time".
    
    Timestamps determine which poses are “going.” The safe turning (and survival turn) is accumulated until the 
    survival threshold is reached, but the total turn is accumulated over the entire run.
    """
    # --- Load poses data ---
    poses = poses_data.get("pose")
    pose_times = poses_data.get("time")
    lanes = poses_data.get("lanes")
    if poses is None or pose_times is None or lanes is None:
        raise ValueError("Poses JSON must contain 'pose', 'time', and 'lanes'.")
    if isinstance(pose_times, str):
        pose_times = ast.literal_eval(pose_times)
    poses = scale_data(poses)
    for i in range(len(lanes)):
        lanes[i] = scale_data(lanes[i])
    
    # --- Load moving data ---
    going_raw = moving_data.get("going")
    moving_times = moving_data.get("time")
    if going_raw is None or moving_times is None:
        raise ValueError("Moving JSON must contain 'going' and 'time'.")
    if isinstance(going_raw, str):
        going_raw = ast.literal_eval(going_raw)
    if isinstance(moving_times, str):
        moving_times = ast.literal_eval(moving_times)
    
    # --- Build "going" intervals from moving.json ---
    intervals = []
    n_moving = len(going_raw)
    if n_moving == 0:
        pass
    elif n_moving == 1:
        if going_raw[0]:
            intervals.append((moving_times[0], pose_times[-1]))
    else:
        i = 0
        while i < n_moving:
            if going_raw[i]:
                start = moving_times[i]
                j = i + 1
                while j < n_moving and going_raw[j]:
                    j += 1
                end = moving_times[j] if j < n_moving else pose_times[-1]
                intervals.append((start, end))
                i = j
            else:
                i += 1

    def is_going(timestamp, intervals):
        for start, end in intervals:
            if start <= timestamp <= end:
                return True
        return False

    # --- Compute centerline and road center ---
    centerline, half_widths = compute_centerline_and_halfwidths(lanes)
    all_points = lanes[0] + lanes[1]
    center_x = np.mean([pt[0] for pt in all_points])
    center_y = np.mean([pt[1] for pt in all_points])
    
    # --- Determine reference angle using the first "going" pose ---
    ref_angle = None
    for i in range(len(poses)):
        if is_going(pose_times[i], intervals):
            x0, y0 = poses[i]
            ref_angle = np.arctan2(y0 - center_y, x0 - center_x)
            break
    if ref_angle is None:
        ref_angle = 0

    # --- Initialize accumulators ---
    survival_turn = None    # Cumulative turning until CTE first exceeds survival_threshold.
    total_turn = 0.0        # Cumulative turning over the entire run.
    safe_turns = {st: 0.0 for st in safe_thresholds}  # Only accumulate until survival threshold.
    cumulative_turn = 0.0   # This will be used similarly as total_turn.
    prev_raw = None
    max_cte = 0.0

    total_speed = 0.0
    speed_count = 0
    prev_going_idx = None

    # --- Process each pose ---
    for i in range(len(poses)):
        if not is_going(pose_times[i], intervals):
            prev_raw = None
            continue

        x, y = poses[i]
        t = pose_times[i]
        curr_cte = compute_computed_cte(x, y, centerline, half_widths)
        max_cte = max(max_cte, curr_cte)

        # Compute turning angle relative to road center.
        current_angle = np.arctan2(y - center_y, x - center_x)
        current_raw = np.degrees(current_angle - ref_angle)
        if prev_raw is None:
            prev_raw = current_raw
            diff = 0.0
        else:
            diff = current_raw - prev_raw
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            prev_raw = current_raw

        # Accumulate total turning over the entire run.
        total_turn += abs(diff)
        # If survival threshold not yet reached, accumulate safe turning.
        if survival_turn is None:
            cumulative_turn += abs(diff)
            for st in safe_thresholds:
                if curr_cte <= st:
                    safe_turns[st] += abs(diff)
            if curr_cte > survival_threshold:
                survival_turn = cumulative_turn

        # Compute speed.
        if prev_going_idx is not None:
            dt = t - pose_times[prev_going_idx]
            if dt > 0:
                x_prev, y_prev = poses[prev_going_idx]
                dist = np.hypot(x - x_prev, y - y_prev)
                total_speed += (dist / dt)
                speed_count += 1
        prev_going_idx = i

    if survival_turn is None:
        survival_turn = cumulative_turn

    safe_percents = {}
    for st in safe_thresholds:
        percent = (safe_turns[st] / survival_turn) * 100 if survival_turn > 0 else 0.0
        safe_percents[st] = min(percent, 100.0)
    mean_speed = (total_speed / speed_count) if speed_count > 0 else 0.0

    return survival_turn, safe_percents, mean_speed, max_cte, total_turn

def main():
    parser = argparse.ArgumentParser(
        description="Generate a performance table from runs using poses.json and moving.json files. "
                    "Computes cumulative turning until computed CTE exceeds the survival threshold (SurvivedDeg), "
                    "as well as the total turning (TotalTurn) over the run. For each safe threshold, computes the "
                    "percentage of turning (up to survival threshold) where CTE was safe. Also computes mean speed, "
                    "maximum CTE, and an overall Score. Runs with TotalTurn < 1080° are skipped. "
                    "Runs are grouped by the subfolder in which they reside."
    )
    parser.add_argument('--folder', type=str,
                        default='/home/cubos98/catkin_ws/src/Vehicle/results_reality',
                        help="Root folder containing dataset folders.")
    parser.add_argument('--dataset', type=str, default='dataset3',
                        help="The dataset folder to process (e.g., dataset3).")
    parser.add_argument('--survival_threshold', type=float, default=4.0,
                        help="The survival CTE threshold (a single float).")
    parser.add_argument('--safe_thresholds', type=str, default='1.5,2.0,3.0',
                        help="Comma-separated list of three safe CTE thresholds.")
    parser.add_argument('--ideal_angle', type=float, default=360.0,
                        help="Ideal turning angle for normalization (default 360).")
    args = parser.parse_args()

    survival_threshold = args.survival_threshold
    safe_thresholds = [float(val.strip()) for val in args.safe_thresholds.split(',') if val.strip()]
    ideal_angle = args.ideal_angle
    dataset_folder = os.path.join(args.folder, args.dataset)

    # --- Recursively find all run files ending with '_poses.json' within the selected dataset folder ---
    runs = {}
    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith('_poses.json'):
                run_name = file.replace("_poses.json", "")
                moving_filename = f"{run_name}_moving.json"
                poses_path = os.path.join(root, file)
                moving_path = os.path.join(root, moving_filename)
                if os.path.exists(moving_path):
                    rel_path = os.path.relpath(root, dataset_folder)
                    group_name = rel_path if rel_path != "." else "default"
                    runs[os.path.join(group_name, run_name)] = {
                        "poses": poses_path, 
                        "moving": moving_path, 
                        "group": group_name
                    }
                else:
                    print(f"Warning: Skipping {run_name} because {moving_filename} not found.")

    results = []
    for run_key, files in runs.items():
        try:
            poses_data = load_data(files["poses"])
            moving_data = load_data(files["moving"])
            surv_turn, safe_percents, mean_speed, max_cte, total_turn = analyze_run(
                poses_data, moving_data, survival_threshold, safe_thresholds
            )
            # Skip runs whose total turning (total_turn) is less than 1080°
            if total_turn < 1080:
                continue

            avg_safe = sum(safe_percents[st] for st in safe_thresholds) / len(safe_thresholds)
            truncated_turn = min(surv_turn, 1080)
            raw_score = (truncated_turn / ideal_angle) * (avg_safe / 100)
            score_perc = (raw_score / 3.0) * 100

            row = {
                "Run": run_key, 
                "MeanSpeed": mean_speed, 
                "MaxCTE": max_cte, 
                "SurvivedDeg": surv_turn, 
                "TotalTurn": total_turn,
                "ScorePerc": score_perc,
                "Group": files["group"]
            }
            for st in safe_thresholds:
                row[f"SafePerc_{st}"] = safe_percents[st]
            results.append(row)
        except Exception as e:
            print(f"Error processing run {run_key}: {e}")

    if not results:
        print("No valid runs found in folder (after filtering for TotalTurn >= 1080°).")
        return

    df = pd.DataFrame(results)
    df = df.sort_values("Run")

    # --- Group results by subfolder (Group) using mean (and max for MaxCTE) and also include the minimum TotalTurn ---
    grouped = df.groupby("Group")
    grouped_df = grouped.agg({
        "MeanSpeed": "mean",
        "MaxCTE": "max",
        "SurvivedDeg": "mean",
        "TotalTurn": "mean",
        "ScorePerc": "mean"
    }).reset_index()
    # Also add the minimum TotalTurn (i.e. the shortest run) for each group.
    min_turn = grouped["TotalTurn"].min().reset_index().rename(columns={"TotalTurn": "MinTotalTurn"})
    grouped_df = pd.merge(grouped_df, min_turn, on="Group")
    for st in safe_thresholds:
        col = f"SafePerc_{st}"
        grouped_mean = grouped[col].mean().reset_index()
        grouped_df = pd.merge(grouped_df, grouped_mean, on="Group", suffixes=("", f"_{col}"))
        grouped_df.rename(columns={f"{col}": col}, inplace=True)

    print("\nIndividual Performance Table:")
    print(df.to_string(index=False))
    print("\nGrouped Performance Table (by subfolder):")
    print(grouped_df.to_string(index=False))
    # Optionally, save to CSV:
    # grouped_df.to_csv("performance_table_grouped.csv", index=False)

if __name__ == '__main__':
    main()
