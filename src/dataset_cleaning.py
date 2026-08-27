import os
import glob
import random
import shutil
import cv2
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Import preprocess_image from preprocessing module
try:
    from src.preprocessing import preprocess_image
except ImportError:
    try:
        from .preprocessing import preprocess_image
    except ImportError:
        from preprocessing import preprocess_image


def split_dataset(clean_root, classes, train_ratio=0.8, seed=42):
    """
    Split processed images in clean_root/<class> into 80/20 train/test folders.
    """
    print("\n--- Performing Train / Test Split (80/20) ---")
    random.seed(seed)

    for cls in classes:
        src = os.path.join(clean_root, cls)
        if not os.path.exists(src):
            continue

        images = [img for img in os.listdir(src) if os.path.isfile(os.path.join(src, img))]
        if not images:
            continue

        random.shuffle(images)
        split = int(len(images) * train_ratio)

        train_dir = os.path.join(clean_root, "train", cls)
        test_dir = os.path.join(clean_root, "test", cls)

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for img in images[:split]:
            shutil.move(os.path.join(src, img), os.path.join(train_dir, img))

        for img in images[split:]:
            shutil.move(os.path.join(src, img), os.path.join(test_dir, img))

        try:
            os.rmdir(src)
        except OSError:
            pass

        print(f"Class '{cls}': {split} train images -> {train_dir}, {len(images) - split} test images -> {test_dir}")


def process_dataset(source_root=None, clean_root=None, max_images_per_class=240, do_split=True, train_ratio=0.8, seed=42, k_clusters=6, dark_spot_dist_threshold=30):
    """
    Loop through dataset classes in source_root (data/), preprocess up to max_images_per_class (default 240)
    images per class using preprocess_image(), save into clean_root (cleaned_data/), and split into
    80/20 train/test directories.
    
    Uses relative paths based on project folder structure:
      - Default source_root: IP_Assignment/data
      - Default clean_root: IP_Assignment/cleaned_data
    """
    # Determine project root relative to this file (src/dataset_cleaning.py -> IP_Assignment/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    if source_root is None:
        source_root = os.path.join(project_root, "data")
    if clean_root is None:
        clean_root = os.path.join(project_root, "cleaned_data")

    classes = ["overripe", "fully_ripe", "unripe"]
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

    print(f"Source Dataset Path : {os.path.abspath(source_root)}")
    print(f"Cleaned Output Path  : {os.path.abspath(clean_root)}")
    print(f"Max Images Per Class : {max_images_per_class}\n")

    for folder in classes:
        print(f"Processing class: {folder}")
        source_dir = os.path.join(source_root, folder)
        clean_dir = os.path.join(clean_root, folder)

        if not os.path.exists(source_dir):
            print(f"  Warning: Source directory '{source_dir}' does not exist. Skipping...")
            continue

        os.makedirs(clean_dir, exist_ok=True)

        img_paths = []
        for ext in extensions:
            img_paths.extend(glob.glob(os.path.join(source_dir, ext)))
        img_paths = sorted(list(set(img_paths)))

        valid_count = 0
        skipped_count = 0

        for img_path in tqdm(img_paths, desc=f"  {folder}"):
            if max_images_per_class is not None and valid_count >= max_images_per_class:
                break

            img_name = os.path.basename(img_path)
            save_path = os.path.join(clean_dir, img_name)

            try:
                image = cv2.imread(img_path)
                if image is None:
                    skipped_count += 1
                    continue

                cleaned_image = preprocess_image(
                    image,
                    k_clusters=k_clusters,
                    dark_spot_dist_threshold=dark_spot_dist_threshold
                )

                cv2.imwrite(save_path, cleaned_image)
                valid_count += 1

            except Exception as e:
                print(f"\n  Failed to process {img_name}: {e}")
                skipped_count += 1

        print(f"  Valid images : {valid_count}")
        print(f"  Skipped      : {skipped_count}\n")

    # Perform 80/20 Train/Test split
    if do_split:
        split_dataset(clean_root, classes, train_ratio=train_ratio, seed=seed)

    print("\n====================")
    print("Dataset Cleaning & Train/Test Split completed!")
    print("====================")


if __name__ == "__main__":
    process_dataset()
