import cv2
import numpy as np
from ultralytics import YOLO
import torch

# Load the YOLOv8 model
model = YOLO("D:\\seeker_algo\\myenv\\Object_detection_YOLO\\Yolo-Weight\\yolov8x.pt")

# Define categories and their tolls
vehicle_tolls = {
    'car': 150,
    'truck': 350,
    'trailer': 500
}

# Initialize counts and total tax
vehicle_count = {'car': 0, 'truck': 0, 'trailer': 0}
total_tax = 0

# Initialize list to store tracked vehicles
tracked_vehicles = []

# Set up video capture
video_path = 'D:\\seeker_algo\\myenv\\Object_detection_YOLO\\Chapter2_YOLO_With_WebCam\\Cars.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Check if video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define the boundary line for vehicle counting
boundary_line_y = 500  # Adjust according to the video's perspective

# Function to check if vehicle crosses the boundary
def check_boundary_crossing(y_center, boundary_line_y):
    return y_center > boundary_line_y

# Function to calculate the Euclidean distance
def calculate_distance(box1, box2):
    # Get the center of both boxes, ensuring they're on the CPU before the calculation
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2

    # Move the values to CPU if they are on GPU (CUDA tensor)
    x1_center = x1_center.cpu().numpy() if isinstance(x1_center, torch.Tensor) else x1_center
    y1_center = y1_center.cpu().numpy() if isinstance(y1_center, torch.Tensor) else y1_center
    x2_center = x2_center.cpu().numpy() if isinstance(x2_center, torch.Tensor) else x2_center
    y2_center = y2_center.cpu().numpy() if isinstance(y2_center, torch.Tensor) else y2_center
    
    return np.sqrt((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2)

# Set a threshold for determining if a vehicle is the same as previously detected
distance_threshold = 62  # Adjust this threshold based on vehicle size and movement
# Initialize video writer to save the output video
output_video_path = 'D:\\seeker_algo\\myenv\\Object_detection_YOLO\\output_video_toll.mp4'  # Set output video path
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Frame capture failed or video ended.")
        break
    
    # Perform detection
    results = model(frame)
    
    # Extract detections (bounding boxes, labels)
    detections = results[0].boxes  # This gives you all the bounding boxes in the frame
    current_vehicles = []
    
    for box in detections:
        # Get box coordinates and class id
        x1, y1, x2, y2 = box.xyxy[0]  # Get bounding box coordinates
        conf = box.conf[0]  # Get confidence score
        cls = box.cls[0]  # Get class label (integer)

        # Convert class to label
        label = model.names[int(cls)]
        
        # Filter for vehicle categories only (car, truck, trailer)
        if label in vehicle_tolls:
            # Get center of the bounding box
            y_center = (y1 + y2) / 2
            
            # Track the vehicle's bounding box
            current_vehicles.append([x1, y1, x2, y2, label])
            
            # Check if the vehicle crosses the boundary line and hasn't been counted yet
            if check_boundary_crossing(y_center, boundary_line_y):
                vehicle_already_counted = False
                
                # Compare the current vehicle with tracked vehicles
                for tracked_vehicle in tracked_vehicles:
                    if tracked_vehicle[4] == label and calculate_distance(tracked_vehicle, [x1, y1, x2, y2]) < distance_threshold:
                        vehicle_already_counted = True
                        break
                
                # If vehicle is not counted before, count it now
                if not vehicle_already_counted:
                    vehicle_count[label] += 1
                    total_tax += vehicle_tolls[label]
                    tracked_vehicles.append([x1, y1, x2, y2, label])  # Track this vehicle
                
                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({vehicle_tolls[label]} Rs)", (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    # Draw the boundary line on the frame
    cv2.line(frame, (0, boundary_line_y), (frame.shape[1], boundary_line_y), (255, 0, 0), 2)
    
    # Show the real-time vehicle count and tax on the frame
    display_text = f"Cars: {vehicle_count['car']} | Trucks: {vehicle_count['truck']} | Trailers: {vehicle_count['trailer']} | Tax: {total_tax} Rs"
    cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # Display the frame
    cv2.imshow('Toll Booth Vehicle Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

# Print final counts and tax
print("Vehicle Count:", vehicle_count)
print("Total Tax Collected:", total_tax, "Rs")
