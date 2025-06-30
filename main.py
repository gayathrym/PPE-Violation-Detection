#!/usr/bin/env python3

# Imports & Configuration
import os
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import pygame
from ultralytics import YOLO

pygame.mixer.init()

# Paths
PERSON_MODEL_PATH    = "models/yolov8l.pt"
EQUIPMENT_MODEL_PATH = "models/best.pt"
VIDEO_PATH           = "40.mp4"
OUTPUT_EXCEL         = "violations_summary.xlsx"
OUTPUT_JSON          = "violations_summary.json"

# Class names & thresholds
CLASS_NAMES = ['Helmet','Goggles','Jacket','Gloves','Footwear']
CUSTOM_THRESHOLDS = {
    'Helmet': 0.65,
    'Goggles': 0.90,
    'Jacket': 0.60,
    'Gloves': 0.80,
    'Footwear': 0.70
}

# Frame size
FRAME_WIDTH, FRAME_HEIGHT = 1020, 500


# Helper Functions

def detect_objects(model, frame, thresholds):
    """Run YOLO inference; filter detections by thresholds."""
    results = model.predict(frame, save=False)
    boxes, scores, classes = [], [], []
    for r in results:
        for box, score, cls in zip(r.boxes.xyxy.cpu().numpy(),
                                   r.boxes.conf.cpu().numpy(),
                                   r.boxes.cls.cpu().numpy()):
            name = CLASS_NAMES[int(cls)]
            if score >= thresholds.get(name, 0.5):
                boxes.append(box)
                scores.append(score)
                classes.append(name)
    return np.array(boxes), scores, classes


def adjust_box(box, reduction=0.2):
    """Shrink box by reduction factor around its center."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx, dy = w * reduction / 2, h * reduction / 2
    return (int(x1 + dx), int(y1 + dy), int(x2 - dx), int(y2 - dy))


def update_missing_items_summary(current_items, summary_list):
    """Merge per-frame missing items into cumulative summary."""
    now = datetime.now().strftime('%H:%M:%S')
    for item in current_items:
        item['Time'] = now
        idx = item['Person'] - 1
        if idx >= len(summary_list):
            summary_list.append(item)
        else:
            existing = summary_list[idx]['Missing Items'].split(", ")
            new = item['Missing Items'].split(", ")
            combined = list(set(existing + new))
            summary_list[idx].update({
                'Missing Items': ", ".join(combined),
                'Time': now
            })


def update_excel(summary, excel_file=OUTPUT_EXCEL, json_file=OUTPUT_JSON):
    """Persist cumulative violations to Excel and JSON."""
    if os.path.exists(excel_file):
        df = pd.read_excel(excel_file)
    else:
        df = pd.DataFrame(columns=['Date','Time','Person','Helmet','Jacket'])
    today = datetime.now().strftime('%Y-%m-%d')
    time_str = datetime.now().strftime('%H:%M:%S')

    # Ensure columns exist
    for col in ['Date','Time','Person','Helmet','Jacket']:
        if col not in df.columns:
            df[col] = np.nan

    for item in summary:
        pid = item['Person']
        missing = item['Missing Items'].split(", ")
        helmet_status = 'No' if 'Helmet' in missing else 'Yes'
        jacket_status = 'No' if 'Jacket' in missing else 'Yes'
        entry = df[
            (df['Date']==today)&
            (df['Time']==time_str)&
            (df['Person']==pid)
        ]
        if not entry.empty:
            idx = entry.index[0]
            df.at[idx,'Helmet'] = helmet_status
            df.at[idx,'Jacket'] = jacket_status
        else:
            new_row = {
                'Date': today, 
                'Time': time_str, 
                'Person': pid,
                'Helmet': helmet_status,
                'Jacket': jacket_status
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_excel(excel_file, index=False)
    # Also save JSON
    with open(json_file, 'w') as f:
        f.write(df.to_json(orient='records', date_format='iso'))


# Entry Point & Main Loop

def main():
    # Load models
    person_model    = YOLO(PERSON_MODEL_PATH)
    equipment_model = YOLO(EQUIPMENT_MODEL_PATH)

    # Video capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    w, h = FRAME_WIDTH, FRAME_HEIGHT

    # Tracking data structures
    alert_playing = False
    missing_items_summary = []
    helmet_detected_frames    = {}
    helmet_not_detected_frames= {}
    jacket_detected_frames    = {}
    jacket_not_detected_frames= {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (w, h))

        # Detect people & equipment
        person_boxes, _, _ = detect_objects(
            person_model, frame, {'Person':0.85})
        equip_boxes, equip_scores, equip_classes = detect_objects(
            equipment_model, frame, CUSTOM_THRESHOLDS)

        person_boxes = person_boxes.astype(int)
        equip_info = [
            (adjust_box(box, 0.5), cls, sc)
            for box, cls, sc in zip(equip_boxes.astype(int), equip_classes, equip_scores)
        ]

        current_frame_missing = []

        for idx, (px1, py1, px2, py2) in enumerate(person_boxes):
            pid = idx + 1
            # Init counters
            helmet_detected_frames.setdefault(pid,0)
            helmet_not_detected_frames.setdefault(pid,0)
            jacket_detected_frames.setdefault(pid,0)
            jacket_not_detected_frames.setdefault(pid,0)

            worn = {'Helmet':False,'Jacket':False}
            for (x1,y1,x2,y2), eq, score in equip_info:
                if eq in worn:
                    if px1 <= x1 <= px2 and py1 <= y1 <= py2 and \
                       px1 <= x2 <= px2 and py1 <= y2 <= py2:
                        worn[eq] = True

            missing = [item for item, ok in worn.items() if not ok]
            # Update counters
            if worn['Helmet']:
                helmet_detected_frames[pid]+=1
            else:
                helmet_not_detected_frames[pid]+=1
            if worn['Jacket']:
                jacket_detected_frames[pid]+=1
            else:
                jacket_not_detected_frames[pid]+=1

            if missing:
                current_frame_missing.append({
                    'Person': pid,
                    'Missing Items': ", ".join(missing)
                })
                color = (0,0,255)
            else:
                color = (0,255,0)

            # Visualize
            cv2.rectangle(frame, (px1,py1),(px2,py2), color, 3)
            label = f"Person {pid}"
            cv2.putText(frame, label, (px1,py1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
            if missing:
                text = "Missing: "+", ".join(missing)
                cv2.putText(frame, text, (px1,py1-40),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

        # Update summaries & outputs
        if current_frame_missing:
            update_missing_items_summary(current_frame_missing, missing_items_summary)
            update_excel(missing_items_summary)

        cv2.imshow("PPE Violation Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
