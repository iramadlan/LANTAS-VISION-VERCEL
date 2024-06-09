
from http.server import BaseHTTPRequestHandler
import cv2
from ultralytics import YOLO, solutions
import gdown
import os
import json

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)

        video_path = data['video_path']
        output_path = 'hasil_object_counting.mp4'
        model_path = '/tmp/LANTAS-VISION.pt'

        # Download model
        drive_link = 'https://drive.google.com/file/d/1j3FV8sq7BqGPU6Z-NInTVCRZif98HT-j/view?usp=sharing'
        download_from_google_drive(drive_link, model_path)

        # Load model
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        line_y = int(h * 0.5)
        line_x_start = int(w * 0.05)
        line_x_end = int(w * 0.95)
        line_points = [(line_x_start, line_y), (line_x_end, line_y)]
        classes_to_count = [0, 1, 2, 3, 4]
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        counter = solutions.ObjectCounter(
            view_img=True,
            reg_pts=line_points,
            classes_names=model.names,
            draw_tracks=True,
            line_thickness=2,
        )

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                break
            tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)
            im0 = counter.start_counting(im0, tracks)
            cv2.line(im0, line_points[0], line_points[1], (255, 0, 255), 2)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'output': output_path}).encode())

def download_from_google_drive(drive_url, output):
    file_id = drive_url.split('/d/')[1].split('/')[0]
    download_url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(download_url, output, quiet=False)
