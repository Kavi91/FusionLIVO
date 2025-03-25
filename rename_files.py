import os
import yaml

def rename_to_10_digits(directory, modality, extension=".png"):
    """Rename files in a modality directory to 10-digit format with specified extension."""
    modality_dir = os.path.join(directory, modality)
    if not os.path.exists(modality_dir):
        print(f"Skipping {modality_dir}: Directory not found.")
        return
    
    files = sorted(os.listdir(modality_dir))
    print(f"Processing {modality_dir}: Found {len(files)} files.")
    
    if len(files) == 0:
        print(f"No files to rename in {modality_dir}.")
        return
    
    for old_name in files:
        base, ext = os.path.splitext(old_name)
        if ext != extension:
            print(f"Skipping file with wrong extension: {old_name} (expected {extension})")
            continue
        if "_" in base:
            base = base.split("_")[-1]  # Handle prefixes if any
        try:
            num = int(base)
            new_name = f"{num:010d}{ext}"
            old_path = os.path.join(modality_dir, old_name)
            new_path = os.path.join(modality_dir, new_name)
            
            if old_name != new_name:
                os.rename(old_path, new_path)
                print(f"Renamed: {old_name} -> {new_name}")
            else:
                print(f"Already in 10-digit format: {old_name}")
        except ValueError:
            print(f"Skipping invalid filename: {old_name} (non-numeric base: {base})")

def rename_modality(config_path, modality_key, extension=".png"):
    """Rename files for a specific modality across all sequences."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    data_dir = cfg['data_dir']
    modality = cfg['modalities'][modality_key]
    all_sequences = (
        cfg['sequences']['train'] +
        cfg['sequences']['val'] +
        cfg['sequences']['test']
    )
    
    for seq in all_sequences:
        seq_dir = os.path.join(data_dir, seq)
        if not os.path.exists(seq_dir):
            print(f"Skipping sequence {seq}: Directory not found.")
            continue
        
        print(f"\nProcessing sequence {seq} for modality {modality_key}")
        rename_to_10_digits(seq_dir, modality, extension)

# Example functions for specific modalities
def rename_depth(config_path):
    """Rename depth .npy files to 10-digit format."""
    rename_modality(config_path, "depth", extension=".npy")

def rename_image_02(config_path):
    """Rename image_02 .png files to 10-digit format."""
    rename_modality(config_path, "left_rgb", extension=".png")

def rename_image_03(config_path):
    """Rename image_03 .png files to 10-digit format."""
    rename_modality(config_path, "right_rgb", extension=".png")

def rename_velodyne(config_path):
    """Rename velodyne .bin files to 10-digit format."""
    rename_modality(config_path, "velodyne", extension=".bin")

if __name__ == "__main__":
    config_path = "config.yaml"
    
    # Call the function for the modality you want to rename
    rename_depth(config_path)  # Rename depth .npy files now
    # Uncomment below to rename other modalities if needed in the future
    # rename_image_02(config_path)
    # rename_image_03(config_path)
    # rename_velodyne(config_path)
    
    print("\nRenaming complete.")