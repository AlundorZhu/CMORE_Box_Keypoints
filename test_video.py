import torch
import cv2
import numpy as np
import argparse
import config


def load_model(model_path: str):
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    return model


def preprocess(frame: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    img = img.astype("float32") / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor


def draw_keypoints(frame: np.ndarray, pred_np: np.ndarray, orig_w: int, orig_h: int):
    visibility_scores = 1.0 / (1.0 + np.exp(-pred_np[:, 2]))
    for idx, (x_norm, y_norm, _) in enumerate(pred_np):
        if visibility_scores[idx] < 0.5:
            continue
        x = int(np.clip(x_norm * orig_w, 0, orig_w - 1))
        y = int(np.clip(y_norm * orig_h, 0, orig_h - 1))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(frame, str(idx), (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)


def main():
    parser = argparse.ArgumentParser(description="Run keypoint model on a video.")
    parser.add_argument("video", help="Path to input video file (or 0 for webcam)")
    parser.add_argument("--model", default="model_traced.pt", help="Path to traced model")
    args = parser.parse_args()

    model = load_model(args.model)

    source = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open video source '{args.video}'")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]
        tensor = preprocess(frame)

        with torch.no_grad():
            preds = model(tensor)

        pred_np = preds.cpu().numpy()[0]  # [num_keypoints, 3]
        draw_keypoints(frame, pred_np, orig_w, orig_h)

        cv2.imshow("Keypoints", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
