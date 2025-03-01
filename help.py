import cv2

# Global variable to store click coordinates
coords = []

def click_event(event, x, y, flags, param):
    """Callback function to capture mouse click coordinates."""
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        coords.append((x, y))
        print(f"Clicked at: ({x}, {y})")

        # Display the clicked point on the image
        img_copy = param.copy()
        cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)  # Draw a red dot
        cv2.imshow("Image", img_copy)

def get_click_coordinates(image_path):
    """Displays an image and waits for the user to click, returning (x, y) coords."""
    global coords

    # Read and display the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found or could not be loaded.")
        return None

    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", click_event, img)

    print("Click anywhere on the image. Press 'q' to exit.")
    cv2.waitKey(0)  # Wait indefinitely for a key press
    cv2.destroyAllWindows()

    return coords  # Return list of clicked coordinates

# Example usage
image_path = "testImage.jpg"  # Replace with the path to your image
click_positions = get_click_coordinates(image_path)
print("All clicked positions:", click_positions)