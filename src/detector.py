# SPDX-License-Identifier: Apache-2.0
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# فقط یک بار ساخته می‌شود
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,
    max_num_faces=1
)

def detect_face(img):

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        raise Exception("هیچ صورتی پیدا نشد.")

    return results.multi_face_landmarks[0]