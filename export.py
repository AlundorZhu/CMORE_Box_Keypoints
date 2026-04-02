import torch
import cv2
import numpy as np
import coremltools as ct
import os
import config
from typing import cast
from model import SingleObjectKeypointDetector, ModelWithNormalization

def main():
    print("Exporting model for deployment...")

    # 1. Instantiate your original model architecture
    original_model = SingleObjectKeypointDetector(num_keypoints=config.NUM_KEYPOINTS)

    # 2. Load your trained weights
    checkpoint_path = 'best_model.pth'  # Adjust this path if your checkpoint is named differently
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using 'python train.py'")
        return

    print(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
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
    print("--- Testing inference with traced model ---")
    
    # Create a dummy image for testing
    # In a real application, you would load an image like this:
    img = cv2.imread("image.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    show_img = img.copy()  # Keep a copy for visualization
    # img = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
    img = img.astype('float32') / 255.0
    tensor_img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        preds = traced_model(tensor_img)
    
    print("Output shape:", preds.shape)
    print("Output predictions (x, y, visibility_logit):")
    print(preds.detach().cpu().numpy())

    # visualize the predicted keypoints on the image
    vis_img = cv2.cvtColor(show_img, cv2.COLOR_RGB2BGR)
    pred_np = preds.detach().cpu().numpy()[0]  # [num_keypoints, 3]
    visibility_scores = 1.0 / (1.0 + np.exp(-pred_np[:, 2]))

    for idx, (x_norm, y_norm, _) in enumerate(pred_np):
        if visibility_scores[idx] < 0.5:
            continue

        x = int(np.clip(x_norm * config.IMG_SIZE, 0, config.IMG_SIZE - 1))
        y = int(np.clip(y_norm * config.IMG_SIZE, 0, config.IMG_SIZE - 1))
        cv2.circle(vis_img, (x, y), 4, (0, 255, 0), -1)
        cv2.putText(vis_img, str(idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    output_image_path = "predicted_keypoints.png"
    cv2.imwrite(output_image_path, vis_img)
    print(f"Saved keypoint visualization to {output_image_path}")
    
    print("--- Test complete ---")

    # 6. Convert to coreml format
    print("--- Converting to CoreML format ---")
    try:
        coreml_input = ct.ImageType(
            name="image",
            shape=(1, 3, config.IMG_SIZE, config.IMG_SIZE),
            color_layout=ct.colorlayout.RGB,
            scale=1.0 / 255.0,
        )
        coreml_model = cast(ct.models.MLModel, ct.convert(
            traced_model,
            inputs=[coreml_input],
            outputs=[ct.TensorType(name="keypoints")],
            minimum_deployment_target=ct.target.iOS18
        ))

        coreml_model.short_description = "Single-object keypoint detector"
        coreml_model.input_description["image"] = (
            f"RGB image normalized to [0, 1], resized to {config.IMG_SIZE}x{config.IMG_SIZE}"
        )
        coreml_model.output_description["keypoints"] = (
            f"Keypoints as [1, {config.NUM_KEYPOINTS}, 3] tensor (x_norm, y_norm, visibility_logit)"
        )

        coreml_path = config.EXPORTED_MODEL_NAME.replace(".pt", ".mlpackage")
        coreml_model.save(coreml_path)
        print(f"Success: Saved CoreML model to {coreml_path}")
    except Exception as e:
        print(f"Error: CoreML conversion failed. Error: {e}")

if __name__ == '__main__':
    main()
