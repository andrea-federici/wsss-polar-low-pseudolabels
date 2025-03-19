import os
import random

import torch


def load_heatmap(base_heatmaps_dir, current_iteration, img_path: str) -> torch.Tensor:
    heatmap_path = os.path.join(
        base_heatmaps_dir, 
        f"iteration_{current_iteration}", 
        os.path.basename(img_path) + ".pt"
    )
    if os.path.exists(heatmap_path):
        print(f"HEATMAP FOUND: {heatmap_path}")
        return torch.load(heatmap_path)
    else:
        print(f"HEATMAP NOT FOUND: {heatmap_path}")
        # Use a random heatmap if the heatmap is not found
        return random_heatmap(base_heatmaps_dir, current_iteration)


# TODO: rewrite this to be more stable
def random_heatmap(base_heatmaps_dir, current_iteration):
    previous_heatmaps_dir = os.path.join(
        base_heatmaps_dir, 
        f"iteration_{current_iteration}"
    ) if current_iteration >= 0 else None # TODO: I dont think I need the if statement. this should always be fine to run
    
    print(f"Previous heatmaps dir: {previous_heatmaps_dir}")
    if previous_heatmaps_dir is not None:
        heatmap_files = [
            os.path.join(previous_heatmaps_dir, f)
            for f in os.listdir(previous_heatmaps_dir) 
            if f.endswith(".pt")
        ]
    else:
        heatmap_files = []
    
    return torch.load(random.choice(heatmap_files))


def load_matching_heatmap(base_heatmaps_dir, iteration, img_path: str) -> torch.Tensor:
    img_basename = os.path.basename(img_path)[:6]  # First 6 characters
    iteration_dir = os.path.join(
        base_heatmaps_dir, 
        f"iteration_{iteration}"
    )

    matching_files = [
        file for file in os.listdir(iteration_dir)
        if file.startswith(img_basename) and file.endswith(".pt")
    ]

    if matching_files:
        heatmap_path = os.path.join(
            iteration_dir,
            random.choice(matching_files)
        )
        print(f"HEATMAP FOUND: {heatmap_path}")
        return torch.load(heatmap_path)
    
    # Fall back to a random heatmap from the iteration directory
    heatmap_files = [
        os.path.join(iteration_dir, file) 
        for file in os.listdir(iteration_dir) if file.endswith(".pt")
    ]
    if heatmap_files:
        print(f"HEATMAP NOT FOUND, but found in iteration dir: {heatmap_path}")
        return torch.load(random.choice(heatmap_files))

    # Final fallback to any random heatmap if no files exist in the 
    # iteration folder
    print(f"HEATMAP NOT FOUND, not even in iteration dir: {heatmap_path}")
    return random_heatmap(base_heatmaps_dir, iteration)


def load_accumulated_heatmap(base_heatmaps_dir, img_path, label, current_iteration):
    # Initialize the accumulated heatmap to all zeros. When the 
    # current image is negative, load_heatmap will not find the 
    # corresponding heatmap, and it will use a random heatmap 
    # instead. This is fine since the size of the heatmap is 
    # always the same.
    accumulated_heatmap = torch.zeros_like(
        load_heatmap(base_heatmaps_dir, current_iteration-1, img_path)
    )

    # Iterate from 0 to current iteration (excluded)
    for it in range(current_iteration):
        if label == 1:  # Positive sample: use its own heatmap
            heatmap = load_heatmap(base_heatmaps_dir, it, img_path)
        else:  # Negative sample: use a random heatmap
            print(f'Trying to find a matching heatmap for {img_path} and iteration {it}')
            heatmap = load_matching_heatmap(base_heatmaps_dir, it, img_path)

        accumulated_heatmap += heatmap
    
    # Normalize the accumulated heatmap to keep values between 0 
    # and 1
    accumulated_heatmap = torch.clamp(accumulated_heatmap, 0, 1)

    return accumulated_heatmap