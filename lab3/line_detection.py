

import cv2
import numpy as np

# Open the default camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to identify the largest white region (workspace)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour (workspace)
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a mask for the workspace
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Restrict processing to only the workspace
        workspace = cv2.bitwise_and(binary, mask)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(workspace, (5, 5), 0)

        # Apply adaptive thresholding to highlight the thick line
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Use morphological operations to remove noise and enhance the thick line
        kernel = np.ones((5, 5), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Detect edges
        edges = cv2.Canny(processed, 50, 150, apertureSize=3)

        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        # Initialize variables to store start and end points of the longest line
        longest_line = None
        max_length = 0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Ensure the line is inside the detected workspace
                if mask[y1, x1] == 255 and mask[y2, x2] == 255:
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # Calculate line length

                    if length > max_length:
                        max_length = length
                        longest_line = (x1, y1, x2, y2)  # Store the longest line

        # If a valid line is detected
        if longest_line is not None:
            x1, y1, x2, y2 = longest_line

            # Draw the detected line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green line

            # Mark start (blue) and end (red) points
            cv2.circle(frame, (x1, y1), 10, (255, 0, 0), -1)  # Blue for start
            cv2.circle(frame, (x2, y2), 10, (0, 0, 255), -1)  # Red for end

            # Compute 10 evenly spaced interpolation points
            num_points = 10
            interpolation_points = np.linspace(0, 1, num_points)

            interpolated_coords = [
                (int(x1 + (x2 - x1) * t), int(y1 + (y2 - y1) * t)) for t in interpolation_points
            ]

            # Mark interpolation points (magenta)
            for px, py in interpolated_coords:
                cv2.circle(frame, (px, py), 5, (255, 0, 255), -1)  # Magenta dots

            # Print coordinates
            print(f"Start Point: ({x1}, {y1})")
            print(f"End Point: ({x2}, {y2})")
            print("Interpolation Points:")
            for i, (px, py) in enumerate(interpolated_coords):
                print(f"Point {i+1}: ({px}, {py})")

    # Display the frame with detected line and points
    cv2.imshow('Processed Line Detection', frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
