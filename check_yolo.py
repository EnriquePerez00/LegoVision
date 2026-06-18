from ultralytics import YOLO
model_lat = YOLO('camara_domo/models/best.pt')
res = model_lat('camara_domo/data/data10/frames/frame_031_lat.jpg')
print("YOLO boxes in frame_031_lat:", res[0].boxes.xyxyn)
