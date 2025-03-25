import os
import numpy as np
import cv2
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_img_stats(config_file):
    try:
        logger.info(f"Loading config file: {config_file}")
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        data_dir = config['data_dir']
        sequences = config['sequences']['train']  # Use training sequences for stats
        img_height = config['img_height']
        img_width = config['img_width']
        seq_len = config['seq_len']
        left_rgb_dir = config['modalities']['left_rgb']
        
        # Collect all image paths
        all_img_paths = []
        for seq in sequences:
            left_dir = os.path.join(data_dir, seq, left_rgb_dir)
            if not os.path.exists(left_dir):
                raise FileNotFoundError(f"Left image directory not found: {left_dir}")
            
            num_frames = len(os.listdir(left_dir))
            logger.info(f"Sequence {seq}: Found {num_frames} frames in {left_dir}")
            
            for i in range(num_frames - seq_len + 1):
                for j in range(seq_len):
                    img_path = os.path.join(left_dir, f"{i+j:010d}.png")
                    all_img_paths.append(img_path)
        
        num_images = len(all_img_paths)
        logger.info(f"Total images to process: {num_images}")
        
        # Compute mean and std in chunks
        chunk_size = 1000
        mean_sum = np.zeros(3, dtype=np.float64)
        std_sum = np.zeros(3, dtype=np.float64)
        count = 0
        
        for i in range(0, num_images, chunk_size):
            chunk_imgs = []
            for j in range(i, min(i + chunk_size, num_images)):
                img = cv2.imread(all_img_paths[j])
                if img is None:
                    raise ValueError(f"Failed to load image: {all_img_paths[j]}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_width, img_height))
                img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
                chunk_imgs.append(img)
            
            chunk_imgs = np.array(chunk_imgs)  # Shape: [chunk_size, 3, height, width]
            mean_sum += np.mean(chunk_imgs, axis=(0, 2, 3)) * chunk_imgs.shape[0]
            std_sum += np.std(chunk_imgs, axis=(0, 2, 3)) * chunk_imgs.shape[0]
            count += chunk_imgs.shape[0]
            logger.info(f"Processed {count}/{num_images} images")
        
        mean = mean_sum / count
        std = std_sum / count
        
        # Save to text files
        norm_path = os.path.join('Norms', f"{config['log_project']}_train")
        logger.info(f"Saving image statistics to {norm_path}")
        os.makedirs(norm_path, exist_ok=True)  # Create the full directory path
        np.savetxt(os.path.join(norm_path, 'img_mean.txt'), mean)
        np.savetxt(os.path.join(norm_path, 'img_std.txt'), std)
        logger.info(f"Saved image statistics to {norm_path}")
    except Exception as e:
        logger.error(f"Error in compute_img_stats: {str(e)}")
        raise

if __name__ == "__main__":
    compute_img_stats("config.yaml")