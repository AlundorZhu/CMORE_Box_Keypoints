import torch
import cv2
import numpy as np
import os
import config
from model import SingleObjectKeypointDetector, ModelWithNormalization

def main():
    print("Exporting model for deployment...")

    # 1. Instantiate your original model architecture
    original_model = SingleObjectKeypointDetector(num_keypoints=config.NUM_KEYPOINTS)

    # 2. Load your trained weights
    checkpoint_path = os.path.join(config.CHECKPOINT_SAVE_DIR, config.BEST_MODEL_NAME)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using 'python train.py'")
        return

    print(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle dictionary-based checkpoints
    if 'model_state_dict' in checkpoint:
        original_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        original_model.load_state_dict(checkpoint)
    
    original_model.eval()

    # 3. Wrap it in the normalization model
    final_model = ModelWithNormalization(original_model)
    final_model.eval()

    # 4. Trace the model with a dummy input
    try:
        dummy_input = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
        traced_model = torch.jit.trace(final_model, dummy_input)
        
        export_path = config.EXPORTED_MODEL_NAME
        traced_model.save(export_path)
        print(f"Success: Saved traced model to {export_path}")
    except Exception as e:
        print(f"Error: Tracing failed. This can happen if the model has dynamic control flow. Error: {e}")
        return

    # 5. (Optional) Test inference with the traced model
    print("
--- Testing inference with traced model ---")
    
    # Create a dummy image for testing
    # In a real application, you would load an image like this:
    # img = cv2.imread("path/to/your/image.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    img = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
    img = img.astype('float32') / 255.0
    tensor_img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        preds = traced_model(tensor_img)
    
    print("Output shape:", preds.shape)
    print("Output predictions (x, y, visibility_logit):")
    print(preds.detach().cpu().numpy())
    print("--- Test complete ---")


if __name__ == '__main__':
    main()
