import os
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Parameters
thresh = 0.25
frame_check = 20
vid_path = r".\sleep"
output_path = r".\combined_output.mp4"

# Load Dlib models
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# VideoWriter initialization (set after first frame is read)
out = None
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v for MP4 format

for vid in os.listdir(vid_path):
    full_path = os.path.join(vid_path, vid)
    cap = cv2.VideoCapture(full_path)
    flag = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        # Set up writer after knowing frame size
        if out is None:
            height, width = frame.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

        if len(subjects) == 0:
            flag += 1
            if flag >= frame_check:
                cv2.putText(frame, "            DRIVER ASLEEP            ", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            for subject in subjects:
                shape = predict(gray, subject)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]

                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                if ear < thresh:
                    flag += 1
                    if flag >= frame_check:
                        cv2.putText(frame, "            DRIVER ASLEEP            ", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    flag = 0

        out.write(frame)  # Save frame to output video

    cap.release()

# Clean up
if out:
    out.release()
cv2.destroyAllWindows()
