import requests
from io import BytesIO
import numpy as np
from PIL import Image
import matplotlib as plt
import cv2

#1)Get Image
    #How do we do that






def main():
    '''
    driver function
    '''
    img2_size = (600,400)
    im2_pts = np.array([[0, 0],[599, 0],[599, 399],[0, 399]])

    pil_image = retreiveImage()
    #image is a PIL object, transfer it to a NumPy object
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    warped = routeUser(image, img2_size, im2_pts)
    
    if warped is None:
        print("No document was detected. Exiting.")
        return

    cv2.imshow('Warped (Press Q to Close)', warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    


def retreiveImage():
    '''
    prompts the user for image url

    returns: image
    '''
    while True:
        #request a url from user
        image_url = input("Enter the image URL: ").strip()

        try:

            #Fetch Image & Check Image Status
            response = requests.get(image_url,timeout=10)
            response.raise_for_status()

            #return the image
            image = Image.open(BytesIO(response.content))
            image.verify()
            return(Image.open(BytesIO(response.content)))
        
        #Check Exception: Network Errors
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
        #Check exception: Error reading Image File
        except (requests.UnidentifiedImageError, OSError):
            print('Invalid Image Format')


def routeUser(image, img2_size, img2_pts):
    '''
    Routes the user to either manual or automatic document detection.
    
    Parameters:
        image: the source image.
        im2_pts: destination points (list of four points) for the warped image.
        
    Returns:
        The warped image.
    '''
    while True:
        try:
            userChoice = int(input("(1) Manual or (2) Automatic mode? "))
        except ValueError:
            print("Please enter a valid number.")
            continue

        if userChoice == 1:
            # Get points from user
            coords = manualPoints(image.copy())
            # Compute the homography matrix using the selected points and given destination points.
            homography = computeH(coords, img2_pts)
            # Determine the output image size from destination points.
            width = img2_size[0]
            height = img2_size[1]
            #Warp Image
            warpedImage = warpImageCV2(image, homography, width, height)
            return warpedImage
        elif userChoice == 2:
            # For automatic mode, you would implement edge detection and contour finding.
            warpedImage = automaticPoints(image)
            return warpedImage
        else:
            print('\n--Incorrect Choice--\n')


def manualPoints(image):
    '''
    onClick function that retrives coordinate points from a passed in image.
    This function will connect with a callback Function to make the onClick functionality work

    paramters: image - image to retrieve coordinate points from
                numOfPoints - number of coordinate points to be collected

    returns: list of list of coordinates that coorespond to user selected points
    '''

    #Create list of coords
    coords = []
    numOfPoints = int(input('How many points are you getting?'))

    #Show image, setup CallBack
    cv2.imshow("Select Points", image)
    cv2.setMouseCallback("Select Points", onClick, (image, coords))

    #Keep Window responsive 
    while(len(coords) < numOfPoints):
        cv2.waitKey(1)


    print("All points Collected:")
    for i in range(numOfPoints):
        print(f"Point {i+1}: {coords[i]}")

    cv2.destroyAllWindows()
    return coords


def onClick(event, x, y, flags, param):
    '''
    callBack functionality for manualPoints function. This function is called whenever the cv2.setMouseCallback line is called.
    This function will append the new coordinates into the list. Once the user is finished, they will click q.

    paramaters: event - type of event detected by openCV
                x - Horizontal Coordinate of the event
                y - Vertical Coordinate of the event
                flags - any extra flags passed by OpenCV
                param - extra parameters passed by the .setMouseCallback function
                        image - image that the event was made on
                        coords - coordinate list for the coords to be stored on

    '''
    # Unpack parameters
    image, coordsList = param

    if event == cv2.EVENT_LBUTTONDOWN:
        coordsList.append([x, y])  # Store coordinates
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Draw a dot
        cv2.imshow("Select Points", image)  # Refresh the image display



def orderPoints(pts):
    """
    Orders a set of four points in the following order:
    top-left, top-right, bottom-right, bottom-left.
    """
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    # The top-left point will have the smallest sum,
    # whereas the bottom-right will have the largest sum.
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # The top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference.
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def automaticPoints(image):
    '''
    Performs Canny Edge Detection on the given image to try and find the document.
    Uses polygon approximation to filter for quadrilaterals and selects the one with the largest area.
    
    Steps:
        1) Convert the image to grayscale.
        2) Apply a Gaussian Blur to reduce noise.
        3) Use the Canny Edge Detector to find edges.
        4) Dilate the edges to make them more robust.
        5) Find contours in the edge map.
        6) For contours with area above a threshold, compute their convex hull and approximate them as a polygon.
        7) If a polygon with 4 vertices is found, assume it’s the document.
        8) Order the 4 points, compute the perspective transform, and warp the image.
    
    Parameters:
        image: Input image (assumed to be in RGB format).
    
    Returns:
        warped: The warped (top-down) view of the document.
                Returns None if no valid document contour is found.
    '''
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian Blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate edges to make them thicker
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours in the edge map
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contourAreaThreshold = 1500
    contouredDocument = None

    # Loop through contours to find a valid quadrilateral
    for contour in contours:
        if cv2.contourArea(contour) < contourAreaThreshold:
            continue
    
        # Calculate the arc length and the convex hull of the contour
        contourArcLength = cv2.arcLength(contour, True)
        contour_hull = cv2.convexHull(contour)
        
        # Approximate the contour to a polygon
        approxImage = cv2.approxPolyDP(contour_hull, 0.007 * contourArcLength, True)
        
        # If the approximated contour has 4 points, assume it's the document
        if len(approxImage) == 4:
            contouredDocument = approxImage
            break

    if contouredDocument is None:
        print("No valid document contour found.")
        return None

    # Order the points in a consistent order: top-left, top-right, bottom-right, bottom-left
    pts = orderPoints(contouredDocument)
    
    # Compute the width of the new image (max distance between points in the horizontal direction)
    widthA = np.linalg.norm(pts[2] - pts[3])
    widthB = np.linalg.norm(pts[1] - pts[0])
    maxWidth = int(max(widthA, widthB))
    
    # Compute the height of the new image (max distance between points in the vertical direction)
    heightA = np.linalg.norm(pts[1] - pts[2])
    heightB = np.linalg.norm(pts[0] - pts[3])
    maxHeight = int(max(heightA, heightB))
    
    # Define destination points for the warped image: top-left, top-right, bottom-right, bottom-left
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped


def computeH(im1_pts, im2_pts):
    """
    Computes a 3x3 homography matrix H that maps points from im1 to im2.
    
    Parameters:
        im1_pts : list or np.ndarray of shape (n, 2)
            Points in the source image.
        im2_pts : list or np.ndarray of shape (n, 2)
            Corresponding points in the destination (warped) image.
            
    Returns:
        H : np.ndarray of shape (3, 3)
            The homography matrix.
    """
    # Ensure the points are numpy arrays of type float
    im1_pts = np.array(im1_pts, dtype=np.float32)
    im2_pts = np.array(im2_pts, dtype=np.float32)
    n = im1_pts.shape[0]
    
    A = []
    for i in range(n):
        x, y = im1_pts[i]
        xp, yp = im2_pts[i]
        row1 = [ x, y, 1, 0, 0, 0, -xp*x, -xp*y, -xp ]
        row2 = [ 0, 0, 0, x, y, 1, -yp*x, -yp*y, -yp ]
        A.append(row1)
        A.append(row2)
        
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]  # solution is the last row of V^T
    H = h.reshape((3, 3))
    
    # Normalize so that H[2,2] == 1 (if possible)
    if np.abs(H[2, 2]) > 1e-12:
        H /= H[2, 2]
    return H

def warpImage(img, H, output_size):
    '''
    Warps the image using backward warping with the given homography.
    
    Parameters:
        img: the source image (BGR format) as a NumPy array.
        H: 3x3 homography matrix mapping source points to destination points.
        output_size: tuple (width, height) for the warped image.
        
    Returns:
        warped_img: the warped image as a NumPy array.
    '''
    width, height = output_size
    # Compute the inverse homography for backward mapping.
    H_inv = np.linalg.inv(H)
    
    # Create grid of destination coordinates.
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones_like(x_coords)
    dest_homog = np.stack([x_coords, y_coords, ones], axis=0)  # shape (3, height, width)
    dest_homog_flat = dest_homog.reshape(3, -1)  # shape (3, num_pixels)
    
    # Map destination coordinates back to the source image.
    src_coords = np.dot(H_inv, dest_homog_flat)
    src_coords /= src_coords[2, :]  # normalize
    
    # Round and cast to integer indices.
    src_x = np.round(src_coords[0, :]).astype(int)
    src_y = np.round(src_coords[1, :]).astype(int)
    
    # Create an empty output image.
    warped_img = np.zeros((height, width, 3), dtype=img.dtype)
    
    # Determine valid coordinates (inside the bounds of the source image).
    valid = (src_x >= 0) & (src_x < img.shape[1]) & (src_y >= 0) & (src_y < img.shape[0])
    
    # Only assign pixels where the source mapping is valid.
    warped_flat = warped_img.reshape(-1, 3)
    warped_flat[valid] = img[src_y[valid], src_x[valid]]
    
    warped_img = warped_flat.reshape(height, width, 3)
    return warped_img

def warpImageCV2(img, H, width, height):
    return cv2.warpPerspective(img, H, (width, height))

if __name__ == '__main__':
    main()