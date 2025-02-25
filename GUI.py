import numpy as np
import cv2
import os
import ctypes
import operator
import screeninfo

class GUI:
    def __init__(self):
        screen = screeninfo.get_monitors()[0]
        taskbar_height = 50 # Adjust if needed based on your environment
        usable_height = screen.height - taskbar_height
        self.screensize = (screen.width, usable_height) # 
        img = np.random.randint(222, size=(self.screensize[1], self.screensize[0], 3),dtype=np.uint8)
        self.canvas = np.array(img, dtype=np.uint8)
        self.canvas_tmp = np.array(img, dtype=np.uint8)
        # self.canvas_w = self.canvas.shape[1] - int(self.screensize[0] * 0.2)
        self.canvas_w = self.screensize[0]
        self.canvas_h = self.screensize[1]
        self.eye_radius = int(0.025 * self.canvas_w)
        self.phase = 0
        self.calibration_cursor_color = (0, 0, 255)
        self.waiting = False
        self.save_pos = False
        self.wait_count = 0
        self.step_w = int(0.025 * self.canvas_w)
        self.step_h = int(0.025 * self.canvas_h)
        self.calibration_cursor_pos = (self.eye_radius, int(0.025 * self.canvas_h))
        self.last_calibration_checkpoint = -1
        self.calibration_counter = 0
        self.offset_y = (self.step_w - self.step_h) if self.step_w > self.step_h else (self.step_h - self.step_w)
        self.calibration_poses = [
            (self.step_w, self.step_h), (20 * self.step_w, self.step_h), (39 * self.step_w, self.step_h),
            (self.step_w, 20 * self.step_h), (20 * self.step_w, 20 * self.step_h), (39 * self.step_w, 20 * self.step_h),
            (self.step_w, 39 * self.step_h), (20 * self.step_w, 39 * self.step_h), (39 * self.step_w, 39 * self.step_h),
            (10 * self.step_w, self.step_h), (30 * self.step_w, self.step_h),
            (10 * self.step_w, 20 * self.step_h), (30 * self.step_w, 20 * self.step_h),
            (10 * self.step_w, 39 * self.step_h), (30 * self.step_w, 39 * self.step_h),
            (self.step_w, 30 * self.step_h), (39 * self.step_w, 10 * self.step_h),
        ]
        self.cursor_radius = 10
        self.cursor_color = (0, 0, 0)
        self.last_cursor = [-1, -1]

        self.consent_given = False # used for privacy reasons -- check function [request_consent]

        self.drawing_mode = False

    def on_trackbar(self, val):
        pass

    def make_window(self, main_image, lateral_images, cursor=None, sensibility=0.95):

        ratio = main_image.shape[1] / main_image.shape[0]

        img = np.random.randint(222, size=(self.screensize[1], self.screensize[0], 3))
        img = np.array(img, dtype=np.uint8)

        main_height = int(self.screensize[1] * 0.8)
        main_width = int(main_height * (main_image.shape[1] / main_image.shape[0]))  # Aspect ratio

        # Main image
        if self.phase == 0:
            main_y_offset = int((self.screensize[1] - main_height) / 3)
            main_x_offset = int((self.screensize[0] - main_width) / 4)
            main_image = cv2.resize(main_image, (main_width, main_height))
            
            img = np.array(self.canvas, copy=True)
            img[main_y_offset:main_image.shape[0] + main_y_offset,
            main_x_offset:main_image.shape[1] + main_x_offset] = main_image
            # Instruction
            img[0:main_y_offset, main_x_offset:main_image.shape[1] + main_x_offset] = cv2.blur(
                img[0:main_y_offset, main_x_offset:main_image.shape[1] + main_x_offset], (10, 10))
            img = cv2.putText(img, 'Show both your eyes to the camera.',
                              (main_x_offset + 10, int(main_y_offset / 4)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255, 255, 255))
        else:
            img = self.canvas

            if self.phase == 2 and not self.drawing_mode:
                img = cv2.copyTo(self.canvas_tmp, None)

        # Lateral Bar
        if self.phase == 1:
            lateral_width = 3 # Narrow bar during calibration
        else:
            lateral_width = 160  # Default width outside calibration

        # img = cv2.putText(img, "Keep still your shoulders and follow the circle with the eyes, moving with your head as more as possibile.",
        #                       (main_x_offset + 10, int(main_y_offset / 2)),
        #                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255, 255, 255))
        
        lateral_height = self.screensize[1]
        img[:, self.screensize[0] - lateral_width:] = (77, 77, 77)

        # Face Zoom Image
        face_frame = lateral_images["face_frame"]
        if face_frame is not None:
            im1_width = int(lateral_width * 0.8)
            im1_height = int(im1_width * (face_frame.shape[0] / face_frame.shape[1]))
            im1_x_offset = int(lateral_width * 0.1)
            face_frame = cv2.resize(face_frame, (im1_width, im1_height))
            img[40:face_frame.shape[0] + 40,
            img.shape[1] - lateral_width + im1_x_offset: img.shape[
                                                             1] - lateral_width + im1_x_offset + im1_width] = \
                face_frame
            img = cv2.putText(img, 'Face', (img.shape[1] - lateral_width + int(lateral_width / 2) - 25, 35),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=(242, 242, 242))

        if self.phase == 0:
            left_eye_frame = lateral_images["left_eye_frame"]
            right_eye_frame = lateral_images["right_eye_frame"]
            # Left Eye Image
            if left_eye_frame is not None:
                im2_width = int(lateral_width * 0.3)
                im2_height = im2_width
                im2_x_offset = int(lateral_width * 0.15)
                im2_y_offset = int(lateral_width * 0.45)
                left_eye_frame = cv2.resize(left_eye_frame, (im2_width, im2_height))
                img[left_eye_frame.shape[0] + 65 + im2_y_offset:2 * left_eye_frame.shape[0] + 65 + im2_y_offset,
                img.shape[1] - lateral_width + im2_x_offset: img.shape[
                                                                 1] - lateral_width + im2_x_offset + im2_width] = \
                    left_eye_frame

            # Right Eye Image
            if right_eye_frame is not None:
                im3_width = int(lateral_width * 0.3)
                im3_height = im3_width  # int(im3_width / ratio)
                im3_x_offset = int(lateral_width * 0.6)
                im3_y_offset = int(lateral_width * 0.45)
                right_eye_frame = cv2.resize(right_eye_frame, (im3_width, im3_height))
                img[right_eye_frame.shape[0] + 65 + im3_y_offset:2 * right_eye_frame.shape[0] + 65 + im3_y_offset,
                img.shape[1] - lateral_width + im3_x_offset: img.shape[
                                                                 1] - lateral_width + im3_x_offset + im3_width] = \
                    right_eye_frame

            if left_eye_frame is not None or right_eye_frame is not None:
                img = cv2.putText(img, 'Eyes',
                                  (img.shape[1] - lateral_width + int(lateral_width / 2) - 25,
                                   face_frame.shape[0] + 105),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=(242, 242, 242))

            # Left Pupil Keypoints Image
            lp_frame = lateral_images["lp_frame"]
            rp_frame = lateral_images["rp_frame"]
            if lp_frame is not None:
                im6_width = int(lateral_width * 0.3)
                im6_height = im6_width  # int(im6_width / ratio)
                im6_x_offset = int(lateral_width * 0.15)
                im6_y_offset = int(lateral_width * 0.45)
                lp_frame = cv2.resize(lp_frame, (im6_width, im6_height))
                img[lp_frame.shape[0] + 165 + im6_y_offset:2 * lp_frame.shape[0] + 165 + im6_y_offset,
                img.shape[1] - lateral_width + im6_x_offset: img.shape[
                                                                 1] - lateral_width + im6_x_offset + im6_width] = \
                    lp_frame

            # Right Pupil Keypoints Image
            if rp_frame is not None:
                im7_width = int(lateral_width * 0.3)
                im7_height = im7_width  # int(im3_width / ratio)
                im7_x_offset = int(lateral_width * 0.6)
                im7_y_offset = int(lateral_width * 0.45)
                rp_frame = cv2.resize(rp_frame, (im7_width, im7_height))
                img[rp_frame.shape[0] + 165 + im7_y_offset:2 * rp_frame.shape[0] + 165 + im7_y_offset,
                img.shape[1] - lateral_width + im7_x_offset: img.shape[
                                                                 1] - lateral_width + im7_x_offset + im7_width] = \
                    rp_frame
            img = cv2.putText(img, "In the next window, keep still your shoulders and follow the circle with the eyes, moving with your head as more as possible.",
                            (main_x_offset + 10, int(main_y_offset / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255, 255, 255))

     
            img = cv2.putText(img, "This application collects eye position data for calibration and interaction processes.",
                            (main_x_offset + 10, int(main_y_offset +25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color=(0, 0, 255),thickness=2)

        if self.phase > 0:
            # Mode
            mode = 'Paint Mode' if self.drawing_mode else 'Pointer Mode'
            if self.phase==1:
                mode = 'Calibration'
            # img = cv2.putText(img, "This application collects eye position data for calibration and interaction processes.", (img.shape[1] - lateral_width + 30, face_frame.shape[0] + 120),
            #       cv2.FONT_HERSHEY_DUPLEX, 0.4, color=col_t)
            col = (111, 111, 111) if self.drawing_mode else (222, 222, 222)
            col_t = (10, 10, 10) if not self.drawing_mode else (255, 255, 255)
            img[face_frame.shape[0] + 60:face_frame.shape[0] + 100,
            img.shape[1] - lateral_width: img.shape[1]] = col
            img = cv2.putText(img, mode, (img.shape[1] - lateral_width + 30, face_frame.shape[0] + 90),
                              cv2.FONT_HERSHEY_DUPLEX, 0.4, color=col_t)  # TODO:migliorare font
            
            

            if cursor is not None and cursor[0] >= 0 and cursor[1] >= 0:
                if not self.drawing_mode:
                    img = cv2.circle(img, (int(cursor[0]), int(cursor[1])), self.cursor_radius, self.cursor_color, -1)
                else:
                    if cursor[0] != self.last_cursor[0] and cursor[1] != self.last_cursor[1]:
                        img = cv2.circle(img, (int(cursor[0]), int(cursor[1])), self.cursor_radius, self.cursor_color,
                                         -1)
                        if self.last_cursor[0] != -1 and self.last_cursor[1] != -1:
                            img = cv2.line(img, (int(cursor[0]), int(cursor[1])),
                                           (int(self.last_cursor[0]), int(self.last_cursor[1])), self.cursor_color,
                                           2 * self.cursor_radius)
                        self.last_cursor = cursor
       
        if self.phase == 2:
            square_dim = int(lateral_width * 0.4)
            text_spacing = 15 # Increased space between square and text
            vertical_gap = 30 # Space between color blocks

            # Test
            letters = ['R', 'G', 'B', 'N', 'Y', 'P', 'A','W/Erase']
            initial_y_offset = face_frame.shape[0] + 100
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 0), (0, 255, 255), (255, 0, 255),
                    (255, 255, 0),(255, 255, 255)]
            for i in range(len(colors)):
                y_start = initial_y_offset + i * (square_dim + vertical_gap)  # Adjust position
                y_end = y_start + square_dim
                
                x_start = img.shape[1] - lateral_width + 40  # Adjust the position in the lateral bar
                x_end = x_start + square_dim
                
                # Draw the color square
                img[y_start:y_end, x_start:x_end] = colors[i]

                # Place the text further below each color square
                img = cv2.putText(img, letters[i], 
                                (x_start + 10, y_end + text_spacing + 10),  # More space added here
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                color=(252, 252, 252), thickness=2)

        cv2.imshow('EyePaint', img)

    def request_consent(self):
        """
        This function was added to display a consent message to the user 
        and wait for their response. If the user does not provide consent,
        the program will not continue.
        """
        message = ("This application collects eye position data for calibration and interaction processes."
                   " Data will not be stored nor shared without your consent."
                   " Do you agree to proceed?")
        response = ctypes.windll.user32.MessageBoxW(
            0, message, "Privacy Notice", 4 # 4 => yes/no dialog box
        )

        if response == 6: # user clicked "yes"
            self.consent_given = True
        else:
            self.consent_given = False
            self.alert_box("Consent Denied", "The application will exit as consent was not provided.")
            exit(0)

    def run_calibration(self):
        # request consent before starting calibration
        # if not self.consent_given:
        #     self.request_consent()
        # # proceeding with calibration if consent is given
        self.canvas[:, :] = (255, 255, 255)
        self.wait_count = 0
        self.calibration_cursor_pos = (15 * self.step_w, 15 * self.step_h + self.offset_y)
        self.step_w *= -1
        self.step_h *= -1

    def calib_step(self, left_visible=False, right_visible=False):
        eyes_visible = left_visible and right_visible
        self.calibration_cursor_color = (0, 255, 0) if eyes_visible else (0, 0, 255)

        if eyes_visible:
            if self.waiting:
                self.wait_count += 1
                self.calibration_cursor_color = (255, 0, 0)
                if self.wait_count == 10:
                    self.wait_count = 0
                    self.waiting = False
                    self.save_pos = False
            else:
                self.calibration_cursor_pos = (
                    list(self.calibration_cursor_pos)[0] + self.step_w,
                    list(self.calibration_cursor_pos)[1] + self.step_h)
                self.check_position()

        self.draw_calibration_canvas()
        self.canvas = cv2.circle(self.canvas, self.calibration_cursor_pos, self.eye_radius,
                                 self.calibration_cursor_color, -1)
        return self.save_pos

    def check_position(self):
        pos_x = int(list(self.calibration_cursor_pos)[0])
        pos_y = int(list(self.calibration_cursor_pos)[1]) - self.offset_y
        pos = (pos_x, pos_y)
        if pos in self.calibration_poses:
            self.save_pos = True if not self.save_pos else False
            self.calibration_cursor_color = (255, 0, 0)
            self.last_calibration_checkpoint += 1
            if not self.waiting:
                self.calibration_counter += 1
                self.waiting = True
            if pos == self.calibration_poses[0]:
                self.step_h = 0
                self.step_w = int(0.025 * self.canvas_w)
            elif pos == self.calibration_poses[2]:
                self.step_h = int(0.025 * self.canvas_h)
                self.step_w = 0
            elif pos == self.calibration_poses[5]:
                self.step_h = 0
                self.step_w = -int(0.025 * self.canvas_w)
            elif pos == self.calibration_poses[3]:
                self.step_h = int(0.025 * self.canvas_h)
                self.step_w = 0
            elif pos == self.calibration_poses[6]:
                self.step_h = 0
                self.step_w = int(0.025 * self.canvas_w)
            elif pos == self.calibration_poses[8]:
                self.step_h = 0
                self.step_w = 0
                self.end_calibration()

    def end_calibration(self):
        self.phase = 2
        self.drawing_mode = False
        self.canvas[:, :] = np.array(np.zeros(self.canvas.shape), dtype=np.uint8)
        self.canvas_tmp[:, :] = (255, 255, 255)

    def toggle_drawing_mode(self):
        self.drawing_mode = not self.drawing_mode
        self.last_cursor = [-1, -1]
        if self.drawing_mode:
            self.canvas = cv2.copyTo(self.canvas_tmp, None)
        else:
            self.canvas_tmp = cv2.copyTo(self.canvas, None)

    def clear_canvas(self):
        self.canvas[:, :] = (255, 255, 255)
        self.canvas_tmp[:, :] = (255, 255, 255)

    def change_cursor_dimension(self, quantity):
        self.cursor_radius += quantity

    def alert_box(self, title, message):
        ctypes.windll.user32.MessageBoxW(0, message, title, 1)

    def check_key(self, k):
        if k == 99:  # c => clear the canvas
            self.clear_canvas()
        elif k == 43:  # + => increase cursor size
            self.change_cursor_dimension(1)
        elif k == 45:  # - => decrease cursor size
            self.change_cursor_dimension(-1)
        elif k == 115:  # s => save the image
            path = os.path.expanduser("~\\Desktop") + "\\painted_with_eyes.png"  # TODO:TOFIX
            cv2.imwrite(path, self.canvas)
            self.alert_box("Image saved", "Image saved correctly in " + path)
        elif k == 114:  # r => RED
            self.cursor_color = (0, 0, 255)
        elif k == 103:  # g => GREEN
            self.cursor_color = (0, 255, 0)
        elif k == 98:  # b => BLUE
            self.cursor_color = (255, 0, 0)
        elif k == 110:  # n => BLACK
            self.cursor_color = (0, 0, 0)
        elif k == 119:  # w => WHITE
            self.cursor_color = (255, 255, 255)
        elif k == 121:  # y = YELLOW
            self.cursor_color = (0, 255, 255)
        elif k == 112:  # p = Fuchsia
            self.cursor_color = (255, 0, 255)
        elif k == 97:  # a = Aqua
            self.cursor_color = (255, 255, 0)

    def draw_calibration_canvas(self):
        self.canvas[:, :] = (255, 255, 255)
        # Draw ghost path
        sp = int(self.cursor_radius)
        checkpoint_poses = [tuple(map(operator.add, e, (sp, sp))) for e in self.calibration_poses]
        self.canvas = cv2.line(self.canvas, checkpoint_poses[0], checkpoint_poses[2], (133, 133, 133),
                               self.cursor_radius)
        self.canvas = cv2.line(self.canvas, checkpoint_poses[3], checkpoint_poses[5], (133, 133, 133),
                               self.cursor_radius)
        self.canvas = cv2.line(self.canvas, checkpoint_poses[6], checkpoint_poses[8], (133, 133, 133),
                               self.cursor_radius)
        self.canvas = cv2.line(self.canvas, checkpoint_poses[2], checkpoint_poses[5], (133, 133, 133),
                               self.cursor_radius)
        self.canvas = cv2.line(self.canvas, checkpoint_poses[3], checkpoint_poses[6], (133, 133, 133),
                               self.cursor_radius)

        checkpoint_color = (111, 111, 111)
        for checkpoint in self.calibration_poses:
            cv2.rectangle(self.canvas, checkpoint, tuple(map(operator.add, checkpoint, (20, 20))), checkpoint_color, -1)

        if self.last_calibration_checkpoint < 0:
            return

        sorted_indices = [0, 9, 1, 10, 2, 16, 5, 12, 4, 11, 3, 15, 6, 13, 7, 14, 8]
        sorted_poses = [self.calibration_poses[idx] for idx in sorted_indices]

        checkpoint_color = (0, 250, 0)
        cv2.rectangle(self.canvas, sorted_poses[0], tuple(map(operator.add, sorted_poses[0], (20, 20))), checkpoint_color,
                      -1)
        for square_idx in range(self.last_calibration_checkpoint):
            prev_square = sorted_poses[square_idx]
            square = sorted_poses[square_idx + 1]
            cv2.rectangle(self.canvas, prev_square, tuple(map(operator.add, prev_square, (20, 20))), checkpoint_color,
                          -1)
            cv2.rectangle(self.canvas, square, tuple(map(operator.add, square, (20, 20))), checkpoint_color, -1)
            self.canvas = cv2.line(self.canvas, tuple(map(operator.add, prev_square, (10, 10))), tuple(map(operator.add, square, (10, 10))), checkpoint_color, self.cursor_radius)
        self.canvas = cv2.line(self.canvas, tuple(map(operator.add, sorted_poses[self.last_calibration_checkpoint], (10, 10))), tuple(map(operator.add, self.calibration_cursor_pos, (10, 10))),
                               checkpoint_color, self.cursor_radius)
