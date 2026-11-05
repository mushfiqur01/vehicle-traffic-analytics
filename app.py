import cv2
import os
import streamlit as st
import supervision as sv
from ultralytics import YOLO

# 1. Dashboard Configuration
st.set_page_config(page_title="Traffic Analytics", layout="wide")
st.title("🚦 Edge-Optimized Traffic Analytics Engine")
st.markdown("**Architecture:** YOLOv8 Nano (ONNX) + ByteTrack Heuristics")

# Ensure uploads directory exists
os.makedirs("uploads", exist_ok=True)

# 2. Sidebar Controls & Drag-and-Drop
st.sidebar.header("📁 Video Source")
uploaded_file = st.sidebar.file_uploader("Upload Traffic Video", type=['mp4', 'avi', 'mov'])

st.sidebar.header("⚙️ Engine Configuration")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)
resolution = st.sidebar.selectbox("Inference Resolution", [320, 480, 640], index=2)

@st.cache_resource
def load_model():
    return YOLO('models/best.onnx', task='detect') 

model = load_model()

if st.button("🚀 Start Live Processing", type="primary"):
    # Determine video source (Uploaded vs Default)
    if uploaded_file is not None:
        video_path = os.path.join("uploads", uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Loaded: {uploaded_file.name}")
    else:
        video_path = 'data/sample_video.mp4'
        st.sidebar.info("Using default sample video.")
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Dynamic counting line based on video size
    START = sv.Point(0, height // 2)
    END = sv.Point(width, height // 2)
    line_zone = sv.LineZone(start=START, end=END)
    
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_scale=1.5)
    
    stframe = st.empty()
    col1, col2 = st.columns(2)
    in_metric = col1.empty()
    out_metric = col2.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Edge Inference
        results = model.track(
            frame, 
            tracker="bytetrack.yaml", 
            persist=True, 
            conf=conf_threshold, 
            imgsz=resolution, 
            verbose=False
        )[0]
        
        if results.boxes is not None and results.boxes.id is not None:
            detections = sv.Detections.from_ultralytics(results)
            
            # --- HEURISTIC FILTER ---
            # Mask out the top 150 pixels to ignore static signboards
            if len(detections) > 0:
                mask = detections.xyxy[:, 1] > 150
                detections = detections[mask]
            
            if len(detections) > 0 and detections.tracker_id is not None:
                line_zone.trigger(detections=detections)
                labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
                
                # Annotate Frame
                try:
                    frame = box_annotator.annotate(scene=frame, detections=detections)
                    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
                except TypeError:
                    frame = box_annotator.annotate(frame=frame, detections=detections)
                    frame = label_annotator.annotate(frame=frame, detections=detections, labels=labels)
        
        try:
            frame = line_annotator.annotate(frame=frame, line_counter=line_zone)
        except TypeError:
            frame = line_annotator.annotate(scene=frame, line_counter=line_zone)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)
        
        in_metric.metric("⏬ Vehicles Inbound", line_zone.in_count)
        out_metric.metric("⏫ Vehicles Outbound", line_zone.out_count)
        
    cap.release()
    
    # Cleanup uploaded file so your hard drive doesn't fill up
    if uploaded_file is not None and os.path.exists(video_path):
        os.remove(video_path)
        
    st.success(f"Processing Complete! Final Count -> In: {line_zone.in_count} | Out: {line_zone.out_count}")