import os
from datasets import load_dataset
from PIL import Image

def main():
    # Load dataset
    print("Loading dataset...")
    # Using the validation split as it's typically smaller and enough to get one image per class
    dataset = load_dataset("Elriggs/imagenet-50-subset", cache_dir="./.data", split="validation")
    
    output_dir = "class_images"
    os.makedirs(output_dir, exist_ok=True)
    
    seen_classes = set()
    
    print(f"Saving images to {output_dir}/ ...")
    
    for item in dataset:
        class_name = item.get("class_name")
        
        # Fallback if class_name is not available
        if class_name is None:
            class_name = str(item.get("label", "unknown"))
            
        if class_name not in seen_classes:
            seen_classes.add(class_name)
            
            image = item["image"]
            
            # Ensure the image is in RGB mode for saving (in case it's in a different format like L/CMYK)
            if hasattr(image, "convert"):
                image = image.convert("RGB")
            
            # Clean up class name for valid file system paths
            safe_class_name = str(class_name).replace("/", "_").replace(" ", "_")
            out_path = os.path.join(output_dir, f"{safe_class_name}.png")
            
            try:
                image.save(out_path)
                print(f"Saved: {out_path} (Class: {class_name})")
            except Exception as e:
                print(f"Failed to save image for class '{class_name}': {e}")
            
        # Stop early when we have 50 classes (since it's imagenet-50-subset)
        if len(seen_classes) >= 50:
            break
            
    print(f"Done! Saved {len(seen_classes)} images representing {len(seen_classes)} unique classes into the '{output_dir}' directory.")

if __name__ == "__main__":
    main()
