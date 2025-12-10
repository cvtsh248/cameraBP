import cv2
import mediapipe as mp
from dataclasses import dataclass
import numpy as np
import math

class DetectorApp():

    def __init__(self):
        pass

    def detectFaces(self, videopath=None):
        cap = None

        if videopath:
            cap = cv2.VideoCapture(videopath)
        else:
            cap = cv2.VideoCapture(0)

        mp_face_mesh_detection = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        with mp_face_mesh_detection.FaceMesh(min_detection_confidence=0.5) as face_mesh:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

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
                        for id, lm in enumerate(face_landmarks.landmark): # Forhead is region of interest
                            if id in forehead_bounds:
                                h, w, c = image.shape
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                forehead_verts = np.append(forehead_verts, np.int32([[cx, cy]]), axis=0)
                        hull = cv2.convexHull(np.int32(forehead_verts))
                        cv2.fillConvexPoly(mask, hull, (255,255,255))

                        filtered_image = cv2.multiply(image.astype(np.float32), mask.astype(np.float32)/255)
                        filtered_image = filtered_image.astype(np.uint8)


                cv2.imshow('MediaPipe Face Detection', filtered_image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = DetectorApp()
    app.detectFaces()