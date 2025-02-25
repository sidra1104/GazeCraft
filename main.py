import cv2
import json
import os
import Detector
import GUI
import Homography
import numpy as np

PUPIL_THRESH = 42
PHASE = 0
# PHASE 0: Pupils configuration
# PHASE 1: Eyes Calibration
# PHASE 2: Paint Mode

cursor_pos = [-1, -1]  # Stores the eye-tracking based cursor position
CALIBRATION_FILE = "calibration_data.json"  # File to save calibration data

def save_calibration_data(data):
    """Save calibration data to a JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    data['calibration_circle_pos'] = data['calibration_circle_pos'].tolist()
    data['calibration_eye_pos'] = data['calibration_eye_pos'].tolist()
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(data, f)

def load_calibration_data():
    """Load calibration data from a JSON file."""
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, 'r') as f:
            data = json.load(f)
        # Convert lists back to numpy arrays
        data['calibration_circle_pos'] = np.array(data['calibration_circle_pos'], dtype=np.float32)
        data['calibration_eye_pos'] = np.array(data['calibration_eye_pos'], dtype=np.float32)
        return data
    return None

if __name__ == '__main__':  # The code under it gets executed when main.py gets executed
    # Creating Objects for detection, GUI, and Homography
    detector = Detector.CascadeDetector()
    gui = GUI.GUI()
    homo = Homography.Homography()

    # Try to load calibration data
    calibration_data = load_calibration_data()
    if calibration_data:
        print("Calibration data loaded successfully.")
        homo.homography = np.array(calibration_data['homography'], dtype=np.float32)
        homo.calibration_circle_pos = calibration_data['calibration_circle_pos']
        homo.calibration_eye_pos = calibration_data['calibration_eye_pos']
    else:
        print("No calibration data found. Proceeding with calibration.")


    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Opens laptop's inbuilt camera
    cv2.namedWindow('EyePaint', cv2.WINDOW_FULLSCREEN)  # Create a main display window
    # Add a trackbar for eye detection threshold and set it to PUPIL_THRESH = 42
    # cv2.createTrackbar('Eye Detection Threshold', 'EyePaint', 0, 255, gui.on_trackbar)
    # cv2.setTrackbarPos('Eye Detection Threshold', 'EyePaint', PUPIL_THRESH)

    while True:
        _, frame = cap.read()  # reading each frame of the video; each frame is an image; _ is a boolean - captured or not
        frame = cv2.flip(frame, 1)  # Mirror the frame along vertical axis because we will be seeing on the screen

        detector.find_eyes(frame)

        if PHASE == 1:
            if homo.homography is not None:

                PHASE = 2
                detector.start_phase(2)
                gui.end_calibration()
            else:
                if gui.calib_step(detector.left_is_visible, detector.right_is_visible):
                    homo.save_calibration_position([detector.left_pupil, detector.right_pupil], gui.calibration_cursor_pos,
                                                   gui.calibration_counter)

                if gui.phase == 2:
                    PHASE = 2
                    homo.calculate_homography()
                    # Save calibration data
                    calibration_data = {
    'homography': homo.homography.tolist() if homo.homography is not None else None,
    'calibration_circle_pos': homo.calibration_circle_pos,
    'calibration_eye_pos': homo.calibration_eye_pos,
}
                    save_calibration_data(calibration_data)
                    print("Calibration data saved.")

                    detector.start_phase(2)   
        elif PHASE == 2:
            cursor_pos = homo.get_cursor_pos([detector.left_pupil, detector.right_pupil])

        gui.make_window(frame, detector.get_images(), cursor_pos, detector.overlap_threshold)

        # TODO disegnare occhi nel face_frame
        k = cv2.waitKey(33)
        if k == 27 | 0xFF == ord('q'):
            break
        elif k == 32:
            if PHASE < 2:
                if PHASE == 0:
                    if not detector.left_is_visible or not detector.right_is_visible:
                        # gui.alert_box("Error", "Show both your eyes to the camera.")
                        detector.phase -= 1
                        gui.phase -= 1
                        PHASE -= 1
                    elif calibration_data:
                        cv2.destroyWindow("EyePaint")
                        cv2.namedWindow('EyePaint', cv2.WINDOW_FULLSCREEN)
                        # cv2.createTrackbar('Eye Detection Threshold', 'EyePaint', 0, 255, gui.on_trackbar)
                        gui.alert_box("Paint Phase", "Keep still your shoulders and move the cursor with your eyes, "
                                                 "changing between drawnig/pointing mode with space key. "
                                                 "Personalize the cursor and change the color by pressing the "
                                                 "relative key on the lateral bar.")
                        gui.run_calibration()
                    else:
                        # gui.alert_box("Calibration Phase", "Keep still your shoulders and follow the circle with "
                        #                                    "the eyes, moving with your head as more as possibile.")
                        cv2.destroyWindow("EyePaint")
                        cv2.namedWindow('EyePaint', cv2.WINDOW_FULLSCREEN)
                        # cv2.createTrackbar('Eye Detection Threshold', 'EyePaint', 0, 255, gui.on_trackbar)
                        gui.run_calibration()
                else:
                    gui.alert_box("Paint Phase", "Keep still your shoulders and move the cursor with your eyes, "
                                                 "changing between drawnig/pointing mode with space key. "
                                                 "Personalize the cursor and change the color by pressing the "
                                                 "relative key on the lateral bar.")
                detector.phase += 1
                gui.phase += 1
                PHASE += 1
            else:
                gui.toggle_drawing_mode()
        elif k == 60:  # < => decrease sensibility
            detector.overlap_threshold -= 0.01
        elif k == 62:  # > => increase sensibility
            detector.overlap_threshold += 0.01
        elif k == 105:  # i => Info on sensibility
            gui.alert_box("Info - Sensibility",
                          "Set the eyes detector sensibility: stop when the purple squares around the eyes are "
                          "stable but also they keep following the eyes smoothly.")

        if PHASE == 1:
            if k == 116:  # t => Info on threshold
                gui.alert_box("Info - Threshold",
                              "Set the pupils detector sensibility: stop when eyes and pupils are stably seen and "
                              "drawn.")

        if PHASE == 2:
            gui.check_key(k)

    cap.release()
    cv2.destroyAllWindows()
