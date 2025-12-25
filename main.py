import cv2
import mediapipe as mp
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class rppgHelper():
    @staticmethod
    def temporalMean(time_interval, pixels):

        mean = np.array([0.,0.,0.], dtype=np.float64)

        # reverse list of pixels to get latest pixel first
        pixels = list(reversed(pixels))

        # time 0
        t0 = pixels[0][1]
        total_frames = 1

        for count, pixel in enumerate(pixels):
            tn = pixel[1]
            if t0 - tn > time_interval: # check if time interval has been crossed
                total_frames = count
                break
            pixel = np.array(pixel[0], dtype=np.float64)
            mean += pixel
            total_frames = count

        mean = mean/total_frames
        return mean
    
            

class DetectorApp():

    def __init__(self):
        self.pixel_buffer = [] # add tuple containing (pixel, timestamp in seconds)
        self.s_buffer = np.array([]).reshape(0,2) # s values
        self.h_buffer = []
        self.time_interval = 1.6 # seconds

    def detectFaces(self, videopath=None):

        cap = None

        if videopath:
            cap = cv2.VideoCapture(videopath)
        else:
            cap = cv2.VideoCapture(0)

        mp_face_mesh_detection = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        temporal_mean = np.array([0.,0.,0.,0.])
        with mp_face_mesh_detection.FaceMesh(min_detection_confidence=0.5) as face_mesh:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = face_mesh.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                filtered_image = image.copy()
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        forehead_bounds = [109, 10, 338, 108, 107, 9, 336, 337]
                        mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
                        forehead_verts = np.empty((0, 2), dtype=np.int32)
                        for id, lm in enumerate(face_landmarks.landmark): # Forehead is region of interest
                            if id in forehead_bounds:
                                h, w, c = image.shape
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                forehead_verts = np.append(forehead_verts, np.int32([[cx, cy]]), axis=0)
                        hull = cv2.convexHull(np.int32(forehead_verts))
                        cv2.fillConvexPoly(mask, hull, (255,255,255))

                        # apply mask
                        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
                        filtered_image = cv2.multiply(image.astype(np.float32), mask.astype(np.float32)/255)
                        filtered_image = filtered_image.astype(np.uint8)
                        
                        # get average color in the forehead region
                        mean_color = cv2.mean(image, mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
                        mean_color = (mean_color[0], mean_color[1], mean_color[2])
                        self.pixel_buffer.append((mean_color, time.time()))

                        # get temporal mean
                        temporal_mean = rppgHelper.temporalMean(self.time_interval, self.pixel_buffer)
                        
                        # temporal normalisation
                        temporal_norm = mean_color/temporal_mean
                        temporal_norm = temporal_norm/np.linalg.norm(temporal_norm) - np.array([1.,1.,1.])
                        # print(temporal_norm)

                        # orthogonal projection of temporal_norm
                        matrix = np.array([[0,1,-1],[-2,1,1]])
                        s = matrix @ temporal_norm
                        if not np.isnan(s).any():
                            self.s_buffer = np.append(self.s_buffer, [s], axis=0)
                        # print(s)

                        # tuning of projection
                        h = s[0] + np.std(self.s_buffer.T[0])/np.std(self.s_buffer.T[1]) * s[1]
                        # print(h)

                        peak_indices, properties = find_peaks(self.h_buffer, prominence=0.01, distance=1)
                        peak_count = len(peak_indices)
                        print(peak_count)

                        if len(self.pixel_buffer) > 2:
                            if self.pixel_buffer[-1][1] - self.pixel_buffer[0][1] > self.time_interval:
                                self.pixel_buffer = []
                                self.s_buffer = np.array([]).reshape(0,2)
                                self.h_buffer = []


                cv2.imshow('MediaPipe Face Detection', filtered_image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = DetectorApp()
    app.detectFaces()