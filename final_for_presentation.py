import cv2
import numpy as np
import time
from collections import deque


def reorder_corners(corners):
    # Sort the points based on their x and y coordinates
    sorted_corners = sorted(corners, key=lambda x: x[0])  # Sort by x coordinate first
    top_left, top_right = sorted_corners[:2]
    bottom_left, bottom_right = sorted_corners[2:]

    # Further sort the top-left and top-right by their y-coordinate
    if top_left[1] > top_right[1]:
        top_left, top_right = top_right, top_left

    if bottom_left[1] > bottom_right[1]:
        bottom_left, bottom_right = bottom_right, bottom_left

    return np.array([top_left, top_right, bottom_left, bottom_right])

def detect_board(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 1)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            approx = reorder_corners(approx.reshape(4, 2))
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(contour)
            if 0.8 <= aspect_ratio <= 1.2 and area > 50000:
                return approx, x, y, w, h
    return None, None, None, None, None


def create_tracker(tracker_type="CSRT"):
    if tracker_type == "CSRT":
        return cv2.TrackerCSRT_create()
    elif tracker_type == "KCF":
        return cv2.TrackerKCF_create()
    elif tracker_type == "MIL":
        return cv2.TrackerMIL_create()
    elif tracker_type == "BOOSTING":
        return cv2.TrackerBoosting_create()
    return None

def draw_bbox_with_area(frame, bbox, text, color=(255, 255, 255)):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, color, 2, 1)
    text_position = (p1[0], max(0, p1[1] - 10))
    cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def is_square(bbox, tolerance=0.2):
    x, y, w, h = bbox
    aspect_ratio = w / h if w > h else h / w
    return 1 - tolerance <= aspect_ratio <= 1 + tolerance

def is_overlap(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)


def is_valid_color(frame, bbox, orange_lower, orange_upper, blue_lower, blue_upper, threshold=0.3):
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    orange_mask = cv2.inRange(hsv_roi, orange_lower, orange_upper)
    blue_mask = cv2.inRange(hsv_roi, blue_lower, blue_upper)
    orange_pixels = cv2.countNonZero(orange_mask)
    blue_pixels = cv2.countNonZero(blue_mask)
    total_pixels = w * h
    return (orange_pixels + blue_pixels) / total_pixels >= threshold


def compute_circle_size_from_board(board):
    distances = [np.linalg.norm(board[i][0] - board[(i+1) % 4][0]) for i in range(4)]
    avg_distance = sum(distances) / len(distances)
    return int(avg_distance / 6)


def stabilize_circles(new_circles, previous_circles, alpha=1, distance_threshold=20):
    if previous_circles is None:
        return new_circles

    stabilized_circles = []
    for new_circle in new_circles:
        x_new, y_new, r_new = new_circle
        matched = False
        for prev_circle in previous_circles:
            x_prev, y_prev, r_prev = prev_circle
            distance = np.sqrt((x_new - x_prev)**2 + (y_new - y_prev)**2)
            if distance < distance_threshold:
                x_stabilized = alpha * x_prev + (1 - alpha) * x_new
                y_stabilized = alpha * y_prev + (1 - alpha) * y_new
                r_stabilized = alpha * r_prev + (1 - alpha) * r_new
                stabilized_circles.append((x_stabilized, y_stabilized, r_stabilized))
                matched = True
                break
        if not matched:
            stabilized_circles.append(new_circle)

    return stabilized_circles

def extract_circle_region(frame, x, y, r):
    """
    Extracts the region of the frame inside a given circle.

    Parameters:
        frame (numpy.ndarray): The video frame.
        x (int): X-coordinate of the circle's center.
        y (int): Y-coordinate of the circle's center.
        r (int): Radius of the circle.

    Returns:
        numpy.ndarray: The cropped circular region.
    """
    # Create a mask with the same size as the frame
    mask = np.zeros_like(frame, dtype=np.uint8)

    # Draw a filled circle on the mask
    cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

    # Apply the mask to the frame
    circular_region = cv2.bitwise_and(frame, mask)

    # Crop the bounding box around the circle
    x_min, y_min = max(0, x - r), max(0, y - r)
    x_max, y_max = min(frame.shape[1], x + r), min(frame.shape[0], y + r)
    cropped_region = circular_region[y_min:y_max, x_min:x_max]

    return cropped_region

def resize_region(region, template):
    """
    Resizes the detected region to the size of the template.
    
    Parameters:
        region (numpy.ndarray): Detected region (animal from video).
        template (numpy.ndarray): Template image to match against.

    Returns:
        numpy.ndarray: Resized detected region.
    """
    return cv2.resize(region, (template.shape[1], template.shape[0]))


def match_templates(region, templates):
    """
    Matches the detected region with each template using template matching.
    
    Parameters:
        region (numpy.ndarray): Detected region (animal from video).
        templates (list): List of template images.

    Returns:
        int: Index of the most similar template.
        float: Matching score of the best match.
    """
    best_match_idx = -1
    best_score = float('-inf')

    for i, template in enumerate(templates):
        resized_region = resize_region(region, template)
        
        # Perform template matching
        result = cv2.matchTemplate(resized_region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        # Keep track of the best match
        if max_val > best_score:
            best_score = max_val
            best_match_idx = i

    return best_match_idx, best_score



def detect_circles_and_dice_in_video(video_path, output_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    previous_circles = None
    start_time = None
    finish_triggered = False
    no_circle_duration = 4
    have_small_dog =0
    have_big_dog =0

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    board, x, y, w, h = detect_board(frame)
    if board is None:
        print("Error: Could not detect the board.")
        cap.release()
        return

    board_corners = board.reshape(4, 2)
    bottom_left = board_corners[1]
    bottom_right = board_corners[3]
    square_side = int(w * 0.23)
    roi_size = (square_side, square_side)
    
    circle_size = compute_circle_size_from_board(board)
    tolerance = 0.07 * circle_size

    orange_lower = np.array([4, 66, 114])
    orange_upper = np.array([15, 252, 253])
    blue_lower = np.array([90, 80, 100])
    blue_upper = np.array([120, 255, 255])


    trackers = []
    tracker_start_times = {}
    
    
    #for traiding:
    buffer = int(max(w,h)*0.15)
    extended_board_rect = (
        x - buffer,
        y - buffer,
        x + w + buffer,
        y + h + buffer
    )

    circle_outside_count = {}
    
    #for animal detection
    templates = [
            cv2.imread("./materials/rabbit.png"),
            cv2.imread("./materials/sheep.png"),
            cv2.imread("./materials/pig.png"),
            cv2.imread("./materials/cow.png"),
            cv2.imread("./materials/horse.png")
            ]

    wolf_template = cv2.imread("./materials/wolf_upright.png", cv2.IMREAD_GRAYSCALE)
    
    #for tracking dog buying
    big_dog = deque(maxlen=40)
    small_dog = deque(maxlen=40)
    switch_dog = deque(maxlen=40)
    switch_wolf_dog = deque(maxlen=60)
    switch_wolf_animals = deque(maxlen=60)
    
    
    # original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # game_fps = cap.get(cv2.CAP_PROP_FPS)
    # output_path = output_path
    # output_video = cv2.VideoWriter(
    #     output_path,
    #     cv2.VideoWriter_fourcc(*"DIVX"),
    #     game_fps,
    #     (original_width, original_height),
    # )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)        
        
    #------------------------------------------- FINISH DETECTION by empty circles
        empty_circle_count = 0
        empty_circles_coordinates =[]
        animals_coordinates = []
        switch = 0
        switch_wolf = 0
        
        empty_circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 
            dp=1.2, minDist=50, 
            param1=200, param2=95, 
            minRadius=int(circle_size - tolerance*4),
            maxRadius=int(circle_size + tolerance*2))
        
        
        if empty_circles is not None:
            empty_circles = np.uint16(np.around(empty_circles))
            for circle in empty_circles[0, :]:
                x, y, r = circle
                empty_circles_coordinates.append((x,y))
                empty_circle_count +=1 
                # Create a circular mask for the detected circle
                mask = np.zeros_like(gray, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)

        if empty_circle_count == 0:
            if start_time is None:
                start_time = time.time()
            elif time.time() - start_time > no_circle_duration:
                finish_triggered = True
        else:
            start_time = None
            finish_triggered = False
            
        if finish_triggered:
            cv2.putText(frame, "Game Won", (400,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    # ---------------------------------------
        #detecting the circles -> animal tokens (also the empty board animals)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
            param1=70, param2=30,
            minRadius=int(circle_size - tolerance),
            maxRadius=int(circle_size + tolerance)
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))[0, :]
            stabilized_circles = stabilize_circles(circles, previous_circles)
            previous_circles = stabilized_circles

            detected_animals=[0, 0, 0, 0, 0]
            
            for circle in stabilized_circles:
                x, y, r = map(int, circle)
                
                skip_circle = False
                for coord_x, coord_y in empty_circles_coordinates:
                    distance = np.sqrt((coord_x - x)**2 + (coord_y - y)**2)
                    if distance <= r+12:
                        skip_circle = True
                        break  # Skip further checks for this circle
                
                if skip_circle:
                    continue
                
            #Determining what animal it is----------------------------------
                animal = extract_circle_region(frame, x, y, r)
                best_match_idx, best_score = match_templates(animal, templates)         
                detected_animals[best_match_idx] += 1
            #--------------------------------------------------------------

                # Draw the circle on the frame
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                animals_coordinates.append((x,y))

                
            #Determining trading --------------               
                #check if the circle is outside the board for trading
                if not (extended_board_rect[0] <= x <= extended_board_rect[2] and
                        extended_board_rect[1] <= y <= extended_board_rect[3]):
                    if (x, y) not in circle_outside_count:
                        circle_outside_count[(x, y)] = 1
                    else:
                        circle_outside_count[(x, y)] += 1

                    if circle_outside_count[(x, y)] > 5:
                        # cv2.rectangle()
                        cv2.putText(frame, "Trading in place", (400, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    if (x, y) in circle_outside_count:
                        del circle_outside_count[(x, y)]
            #-------------------------------
        
        #printing status of the game---------------
            cv2.rectangle(frame, (8, 8), (130, 180), (255, 255, 255), -1)
            cv2.putText(frame, f"Rabbits: {detected_animals[0]}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, f"Sheeps: {detected_animals[1]}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, f"Pigs: {detected_animals[2]}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, f"Cows: {detected_animals[3]}", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, f"Horses: {detected_animals[4]}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        #------------------------------------------
    #Atack of wolf based on animals dissapearance 
        if 15- empty_circle_count > 0:
            apending =1
        else :
            apending =0
        switch_wolf_animals.append(apending)
        if empty_circle_count == 15:
            if sum(switch_wolf_animals) > 20 :
                cv2.putText(frame, "Wolf attacked", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

             
        

        cv2.polylines(frame, [board[[2,0,1,3]]], isClosed=True, color=(255, 0, 255), thickness=4)

        roi_bottom_right = (
            int(bottom_left[0] + (square_side//4)),
            int(bottom_left[1] - (square_side//4) - roi_size[1]),
            roi_size[0],
            roi_size[1]
        )
        roi_bottom_left = (
            int(bottom_right[0] - (square_side//7)- roi_size[0]),
            int(bottom_right[1] - (square_side//7) - roi_size[1]),
            roi_size[0],
            roi_size[1]
        )


#detect dog as circles, determine dog boxes, then frames it is in 
        roi_frame_right = frame[roi_bottom_right[1]:roi_bottom_right[1] + roi_bottom_right[3], 
                               roi_bottom_right[0]:roi_bottom_right[0] + roi_bottom_right[2]]
        roi_frame_left = frame[roi_bottom_left[1]:roi_bottom_left[1] + roi_bottom_left[3], 
                                roi_bottom_left[0]:roi_bottom_left[0] + roi_bottom_left[2]]

        roi_gray_left = cv2.cvtColor(roi_frame_left, cv2.COLOR_BGR2GRAY)
        roi_gray_right = cv2.cvtColor(roi_frame_right, cv2.COLOR_BGR2GRAY)
        l=cv2.GaussianBlur(roi_gray_left, (5, 5), 1.4)
        r=cv2.GaussianBlur(roi_gray_right, (5, 5), 1.4)
        
        minradius = int(circle_size//4)
        circles_left = cv2.HoughCircles(l, cv2.HOUGH_GRADIENT, 1, roi_size[0] / 4,
                                         param1=50, param2=20, minRadius=minradius, maxRadius=50)

        circles_right = cv2.HoughCircles(r, cv2.HOUGH_GRADIENT, 1, roi_size[0] / 4,
                                         param1=50, param2=20, minRadius=minradius, maxRadius=50)
        
        if circles_left is not None:
            small_dog.append(True)
        else:
             small_dog.append(False)
                  
        if circles_right is not None:
            big_dog.append(True)
        else:
            big_dog.append(False)
            
        small_count = sum(small_dog)
        big_count = sum(big_dog)
        if small_count > (len(small_dog) - small_count) //2 : 
            switch = 1 if have_small_dog == False else 0
            have_small_dog = True
        else:
            have_small_dog = False
            
        if big_count > (len(big_dog) - big_count)//2 : 
            switch = 1 if have_big_dog == False else 0
            have_big_dog = True
        else:
            switch_wolf = 1 if have_big_dog == True else 0
            have_big_dog = False
            
        switch_dog.append(switch)
        switch_wolf_dog.append(switch_wolf)

        #Show having dogs
        cv2.putText(frame, f"Small dog: {have_small_dog}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (32, 50, 92), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Big dog: {have_big_dog}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (32, 50, 92), 2, cv2.LINE_AA)
        if sum(switch_dog) >0:
            cv2.putText(frame, "Got a dog", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #Show wolf if dog dissapears
        if sum(switch_wolf_dog) > 0:
            cv2.putText(frame, "Wolf attacked", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  
# Detect dice
        #Not detecting inside board
        frame_with_border_masked = frame.copy()
        cv2.rectangle(frame_with_border_masked, board[0], 
                      board[3], (0, 0, 0), -1)
        
        hsv_frame = cv2.cvtColor(frame_with_border_masked, cv2.COLOR_BGR2HSV)
        orange_mask = cv2.inRange(hsv_frame, orange_lower, orange_upper)
        blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)
        combined_mask = cv2.bitwise_or(orange_mask, blue_mask)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 4500:
                x, y, w, h = cv2.boundingRect(contour)
                
                #for the case when pigs are dices in trading
                skip_dice = False
                for coord_x, coord_y in animals_coordinates:
                    distance = np.sqrt((coord_x - x)**2 + (coord_y - y)**2)
                    if distance <= w:
                        skip_dice = True
                        break  # Skip further checks for this circle
                
                if skip_dice:
                    continue
                
                if is_square((x, y, w, h)):
                    detected_bboxes.append((x, y, w, h, area))

        current_time = time.time()
#trackers for dice         
        for bbox_data in detected_bboxes:
            bbox = bbox_data[:4]
            is_new = True
            for tracker, tracked_bbox in trackers:
                if is_overlap(bbox, tracked_bbox):
                    is_new = False
                    break

            if is_new and is_valid_color(frame, bbox, orange_lower, orange_upper, blue_lower, blue_upper):
                tracker = create_tracker("KCF")
                tracker.init(frame, bbox)
                trackers.append((tracker, bbox))
                tracker_start_times[tracker] = current_time

        for i, (tracker, _) in enumerate(trackers):
            ok, bbox = tracker.update(frame)
            if ok:
                if is_valid_color(frame, bbox, orange_lower, orange_upper, blue_lower, blue_upper):
                    trackers[i] = (tracker, bbox)
                    area = bbox[2] * bbox[3]
                    draw_bbox_with_area(frame, bbox, "Dice", (0, 255, 0))
                    
#Dice rolled event trigger-------------------------------
                    if current_time - tracker_start_times[tracker] >= 2:
                        text_position = (bbox[0], bbox[1] - 10)
                        cv2.putText(frame, "Dice Rolled", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                else:
                    trackers[i] = None
            else:
                trackers[i] = None

        trackers = [t for t in trackers if t is not None]
        # output_video.write(frame)
        
        cv2.imshow("Circles and Dice in Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # output_video.release()
    cv2.destroyAllWindows()


# ## WITHOUT SAVING
# detect_circles_and_dice_in_video('./materials/E1.MP4')
# detect_circles_and_dice_in_video('./materials/E1-2.MP4')
# detect_circles_and_dice_in_video('./materials/E2.MP4')
# detect_circles_and_dice_in_video('./materials/E3.MP4')

detect_circles_and_dice_in_video('./materials/M1.MP4')
# detect_circles_and_dice_in_video('./materials/M2.MP4')
# detect_circles_and_dice_in_video('./materials/M3.MP4')

# detect_circles_and_dice_in_video('./materials/H1-better.MP4')
# detect_circles_and_dice_in_video('./materials/H1.MP4')
# detect_circles_and_dice_in_video('./materials/H2.MP4')
# detect_circles_and_dice_in_video('./materials/H3.MP4')
