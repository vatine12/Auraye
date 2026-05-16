import cv2
import time
import subprocess
import numpy as np
import threading
from ultralytics import YOLO




# 1. Camera settings
VIDEO_PATH = 0
CAM_WIDTH = 640
CAM_HEIGHT = 240
TARGET_FPS = 60


BASELINE = 0.062
FOCAL_LENGTH = 185


DIST_PERSON_LIMIT = 3.0  
DIST_BIKE_LIMIT = 3.0    
DIST_OBSTACLE_LIMIT = 0.5




MAX_DEPTH_READING = 15.0




# Region of interest
L_ROI_START, L_ROI_END = 0.15, 0.33
C_ROI_START, C_ROI_END = 0.35, 0.65
R_ROI_START, R_ROI_END = 0.70, 0.98
ROI_Y_START, ROI_Y_END = 0.20, 0.70


CAM_FOV_DEG = 120.0




# 3. Yolo settings
MODEL_PATH = "model_yolov8n.tflite"
DETECT_CLASSES = [0, 1]
YOLO_IMGSZ = 320




# 4. Audio settings
BEEP_PITCH = 1000
BEEP_DURATION = 0.05
SAMPLE_RATE = 44100
SLOWEST_INTERVAL = 0.8
FASTEST_INTERVAL = 0.1
VOLUME = 0.5




# 5. Shared global state
GLOBAL_STATE = {
   "frame": None,
   "yolo_results": None,
   "threat_score": 0.0,    
   "pan": 0.5,
   "running": True,
   "new_frame_ready": False,
   "detected_type": None,  
   "clock_pos": None,      
   "distance": 0.0,
   "instruction_needed": False,
   "is_speaking": False,      
}




# 6. Threaded yolo
def yolo_worker():
   print(" Thread started...")
   try:
       model = YOLO(MODEL_PATH, task='detect')
   except Exception as e:
       print(f" YOLO Error: {e}")
       return




   while GLOBAL_STATE["running"]:
       if not GLOBAL_STATE["new_frame_ready"] or GLOBAL_STATE["frame"] is None:
           time.sleep(0.005)
           continue




       img_input = GLOBAL_STATE["frame"].copy()
       GLOBAL_STATE["new_frame_ready"] = False




       h, w = img_input.shape[:2]
       scale_x = w / YOLO_IMGSZ
       yolo_input = cv2.resize(img_input, (YOLO_IMGSZ, int(h / scale_x)))




       try:
           results = model.track(
               yolo_input, imgsz=YOLO_IMGSZ, conf=0.40,
               classes=DETECT_CLASSES, persist=True, verbose=False,
               tracker="bytetrack.yaml"
           )[0]
           GLOBAL_STATE["yolo_results"] = (results, scale_x)
       except Exception:
           pass
ai_thread = threading.Thread(target=yolo_worker, daemon=True)
ai_thread.start()




# 7. Audio engine
def audio_engine_loop():
   t = np.linspace(0, BEEP_DURATION, int(SAMPLE_RATE * BEEP_DURATION), False)
   BASE_WAVE = np.sin(2 * np.pi * BEEP_PITCH * t) * 32767
   last_beep_time = 0.0




   while GLOBAL_STATE["running"]:
       if GLOBAL_STATE["is_speaking"]:
           time.sleep(0.1)
           continue




       threat = GLOBAL_STATE["threat_score"]




       if threat <= 0.0:
           time.sleep(0.05)
           continue




       interval = SLOWEST_INTERVAL - (threat * (SLOWEST_INTERVAL - FASTEST_INTERVAL))
       interval = max(FASTEST_INTERVAL, interval)
       now = time.time()
       if now - last_beep_time > interval:
           try:
               pan = np.clip(GLOBAL_STATE["pan"], 0.0, 1.0)
               L = (BASE_WAVE * (1.0 - pan) * VOLUME).astype(np.int16)
               R = (BASE_WAVE * pan * VOLUME).astype(np.int16)
               data = np.dstack((L, R)).flatten().tobytes()




               subprocess.Popen(
                   ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "2"],
                   stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
               ).communicate(input=data)
           except Exception:
               pass
           last_beep_time = now
       time.sleep(0.01)




audio_thread = threading.Thread(target=audio_engine_loop, daemon=True)
audio_thread.start()




# 8. Voice command
def clock_voice_loop():
   last_spoken_time = 0.0
   COOLDOWN = 5.0




   while GLOBAL_STATE["running"]:
       needed = GLOBAL_STATE["instruction_needed"]
       now = time.time()




       if needed and (now - last_spoken_time > COOLDOWN):
           GLOBAL_STATE["is_speaking"] = True




           obj = GLOBAL_STATE["detected_type"]
           clk = GLOBAL_STATE["clock_pos"]    




           if clk:
               hour = clk.split()[0]
               phrase = f"{obj} at {hour}"
           else:
               phrase = "Watch out"




           print(f" COMMAND: {phrase}")




           try:
               subprocess.run(
                   ["espeak", "-s", "170", phrase],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
               )
           except Exception:
               pass




           GLOBAL_STATE["is_speaking"] = False
           last_spoken_time = now




       time.sleep(0.1)




clock_thread = threading.Thread(target=clock_voice_loop, daemon=True)
clock_thread.start()




# 9. Helper functions
stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15)
stereo.setMinDisparity(0)
stereo.setUniquenessRatio(15)




def get_depth(disparity, x1, x2, y1, y2, percentile=50):
   h, w = disparity.shape
   x1, x2 = max(0, x1), min(w, x2)
   y1, y2 = max(0, y1), min(h, y2)
   roi = disparity[y1:y2, x1:x2]
   valid = roi[roi > 0]
   if valid.size == 0: return MAX_DEPTH_READING
   disp_val = np.percentile(valid, 100 - percentile)
   if disp_val <= 0: return MAX_DEPTH_READING
   return (FOCAL_LENGTH * BASELINE) / disp_val




def angle_to_clock(angle_deg):
   half_fov = CAM_FOV_DEG / 2.0
   angle_deg = max(-half_fov, min(half_fov, angle_deg))
   if angle_deg <= -45: return "10 o'clock"
   elif angle_deg <= -15: return "11 o'clock"
   elif angle_deg < 15: return "12 o'clock"
   elif angle_deg < 45: return "1 o'clock"
   else: return "2 o'clock"




# 10. Main loop
def main():
   print(" System active, press Ctrl+C to stop.")




   # Initialization
   cap = cv2.VideoCapture(VIDEO_PATH)
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
 
   try:
       while GLOBAL_STATE["running"]:
           # 1. read frame
           ret, frame = cap.read()
         
           # Auto-reboot
           if not ret:
               print("!!! Camera signal lost. Waiting for reconnection... !!!", end="\r")
             
               # Silence alarms while no camera
               GLOBAL_STATE["threat_score"] = 0.0
               GLOBAL_STATE["instruction_needed"] = False
             
               # 2. Release broken camera handle
               cap.release()
             
               # 3. Wait a moment
               time.sleep(2.0)
             
               # 4. Attempt to reconnect
               try:
                   cap = cv2.VideoCapture(VIDEO_PATH)
                   if cap.isOpened():
                       cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
                       cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
                       print("\n Camera reconnected! Resuming...")
               except Exception:
                   pass
             
               continue
           # ------------




           h, w, _ = frame.shape
           half_w = w // 2
           img_left = frame[:, :half_w]
           img_right = frame[:, half_w:]




           if not GLOBAL_STATE["new_frame_ready"]:
               GLOBAL_STATE["frame"] = img_left
               GLOBAL_STATE["new_frame_ready"] = True




           try:
               grayL = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
               grayR = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
               disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
           except Exception:
               continue




           curr_threat = 0.0
           curr_pan = 0.5
           curr_type = None
           curr_clock = None
           should_instruct = False




           # Process YOLO results
           res_data = GLOBAL_STATE["yolo_results"]
           if res_data:
               results, scale_x = res_data
               if results.boxes is not None:
                   for box in results.boxes:
                       x1, y1, x2, y2 = (box.xyxy[0].cpu().numpy() * scale_x).astype(int)
                       cx = (x1 + x2) // 2
                       cy = (y1 + y2) // 2
                       cls_id = int(box.cls[0].item())




                       obj_dist = get_depth(disparity, cx-10, cx+10, cy-10, cy+10)




                       if cls_id == 0:
                           otype = "person"
                           limit = DIST_PERSON_LIMIT
                       else:
                           otype = "bicycle"
                           limit = DIST_BIKE_LIMIT




                       norm = (cx - (half_w/2.0)) / (half_w/2.0)
                       clock = angle_to_clock(norm * (CAM_FOV_DEG/2.0))




                       if obj_dist < MAX_DEPTH_READING:
                           if obj_dist <= limit:
                               t = 0.5 + (0.5 * (1.0 - (obj_dist / limit)))
                               curr_threat = max(curr_threat, t)
                               should_instruct = False
                           else:
                               curr_threat = max(curr_threat, 0.3)
                               should_instruct = True
                         
                           curr_type = otype
                           curr_clock = clock
                           curr_pan = cx / float(half_w)




                           color = (0, 0, 255) if obj_dist <= limit else (0, 255, 255)
                           cv2.rectangle(img_left, (x1, y1), (x2, y2), color, 2)
                           cv2.putText(img_left, f"{obj_dist:.1f}m {otype}", (x1, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)




           # Process obstacles
           y1_roi, y2_roi = int(h * ROI_Y_START), int(h * ROI_Y_END)
           rois = [
               (int(half_w*L_ROI_START), int(half_w*L_ROI_END), "11 o'clock", 0.2),
               (int(half_w*C_ROI_START), int(half_w*C_ROI_END), "12 o'clock", 0.5),
               (int(half_w*R_ROI_START), int(half_w*R_ROI_END), "1 o'clock", 0.8)
           ]




           for (rx1, rx2, rclk, rpan) in rois:
               d = get_depth(disparity, rx1, rx2, y1_roi, y2_roi, percentile=30)
               if d <= DIST_OBSTACLE_LIMIT:
                   curr_threat = 1.0
                   curr_pan = rpan
                   curr_type = "obstacle"
                   curr_clock = rclk
                   should_instruct = False
                   cv2.rectangle(img_left, (rx1, y1_roi), (rx2, y2_roi), (0, 0, 255), 2)
                         
           GLOBAL_STATE["threat_score"] = min(curr_threat, 1.0)
           GLOBAL_STATE["pan"] = curr_pan
           GLOBAL_STATE["detected_type"] = curr_type
           GLOBAL_STATE["clock_pos"] = curr_clock
           GLOBAL_STATE["instruction_needed"] = should_instruct




           status = "BEEPING" if not should_instruct and curr_threat > 0 else "INSTRUCTING" if should_instruct else "SCANNING"
           cv2.putText(img_left, f"MODE: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
           cv2.imshow("Smart Guide", img_left)




           if cv2.waitKey(1) & 0xFF == ord('q'):
               break




   except KeyboardInterrupt:
       print("\n User interrupted. Stopping...")
   finally:
       GLOBAL_STATE["running"] = False
       cap.release()
       cv2.destroyAllWindows()




if __name__ == "__main__":
   main()