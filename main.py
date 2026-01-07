import cv2
from ultralytics import YOLO
import torch

# --- CONFIGURATION ---
VIDEO_PATH = "video.mp4"       # Path to the input video file
OUTPUT_PATH = "result.mp4"     # Path for the output video
MODEL_NAME = "yolov8n.pt"      # YOLO model to use (nano is fastest)
CONFIDENCE_THRESHOLD = 0.5     # Ignore detections with low confidence
LINE_POSITION_RATIO = 0.6      # Line position (0.6 means 60% down from top)
OFFSET = 6                     # Pixel tolerance for line crossing

def main():
    # 1. Device Selection (Use GPU if available for speed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] System running on: {device.upper()}")

    # 2. Load the Model
    print(f"[INFO] Loading YOLOv8 model ({MODEL_NAME})...")
    model = YOLO(MODEL_NAME)
    model.to(device)

    # 3. Open Video Source
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: '{VIDEO_PATH}'")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize Video Writer (MP4 format)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    # Calculate Line Y-Coordinate
    line_y = int(height * LINE_POSITION_RATIO)

    # Counter Variables
    counter = 0
    counted_ids = set() # Set to store IDs of objects already counted

    print("[INFO] Processing started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # 4. YOLOv8 Tracking
        # persist=True: Keeps track of IDs across frames
        results = model.track(frame, persist=True, verbose=False, conf=CONFIDENCE_THRESHOLD)
        
        # Process detections
        if results[0].boxes.id is not None:
            # Get boxes, IDs, and class indices
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # Calculate Center Point of the object
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Draw Bounding Box & ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw Center Point (Red Dot)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                # --- COUNTING LOGIC (Line Crossing) ---
                # Check if the center point is within the OFFSET range of the line
                if (line_y - OFFSET) < cy < (line_y + OFFSET):
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        counter += 1
                        # Visual effect: Change line color when counted
                        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 4)
                        print(f"[EVENT] Object ID {track_id} counted! Total: {counter}")

        # 5. Visualization (UI Polish)
        
        # Draw the Counting Line
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)

        # Display Counter on Screen
        text = f"Total Count: {counter}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        
        # Draw background rectangle for text (Better readability)
        cv2.rectangle(frame, (15, 15), (20 + text_w, 60), (0, 0, 0), -1)
        # Draw text
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Save and Show Frame
        out.write(frame)
        cv2.imshow("AI Counter - Portfolio Project", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Process completed. Video saved as '{OUTPUT_PATH}'.")

if __name__ == "__main__":
    main()