import requests
from io import BytesIO
import numpy as np
from PIL import Image, UnidentifiedImageError
import matplotlib as plt
import cv2

#The manual Homography/Warping worked with:
#https://media.istockphoto.com/id/1943833207/photo/woman-playing-in-club-soccer-tournament-giving-daughter-a-hug-on-the-sidelines.jpg?s=612x612&w=is&k=20&c=YAdxq5zrXzZOBD20KefY9bquqadwQ-3emNbp0BHvXvE=
#https://i.redd.it/bebi64i3kbb31.jpg

'''
Previous: 
Explain the idea behind your method:
The drive function is main() of course. This routes the user to retreive Image where I use requests and PIL to request a URL for an image from the user, validates it, then returns it the pillow image object.
Then in main i transfer the pillow object into the a numpy object for data manipulation using openCV. The user is then ruoted to ruoteUser where they are asked if they wante to use Automatic document detection or Manual point selection.

If the user selects Manual the image from the URL appears and the user is prompted to select the points. The coordinates they chose are saved in a list for future manipulation.
The coordinates from the user clicks and the new images corner coordinates are then sent to a Homography function that returns a Homography matrix to warp the image using the users selected points.
The image is then warped using the created Homography matrix and displayed to the user until they click q and terminate the program.

If the user selects Automatic they are not prompted for anything.
The image is sent from main to the automaticPoints function. In here, I apply a canny edge detector to the image to try and find the contours of the document inside the image. 
I then loop through all the found contours, approximate their areas as a polygon, and only consider those contours that have areas > thresholdArea.
If the contour has 4 points, i assume it to be my inlaid document and break out of my contour loop.
I then check my saved contoured document and apply K-means to help clear up noise.

I was having a lot of trouble with the automatic detection of the document.
The first issue I had was only small contours was being detected so I had to figure out a way to filter out the small contours.
This is when I added a loop and a threshold area size for to try and find the best contour.
After that the contour being returned was warping and returning an all black image. I was very confused by this and how to fix it however I noticed that the dimensions where dynamic so it was returning dynamic contours on the image.
I could not figure out how to fix this unfortanately, when trying with a new image it did not do the same thing so I am confused why one image is returning black and the other is not.

One way to improve would be better edge detection. Edge detection to tell the foreground and background apart would be the best way that you can create your contours allowing you to segment your image and creat your warped images.
Another way would be to create fine tuning algorithms for the hyperparamters such as the thresholds which would tune for each image. This would allow for the algorithm to be dynamic for each image given to it.

-----------------------------------------------------------------------------------------------------------------------------------------------------

New:
Converted the points in the order points to order them based on cartesian coordinates due to the way I had the img2 points set up.
I had the img2 points setup as cartesian however the openCV points is not naturally cartesian so it was not transferring correctly.

I was using order points in the automatic... I dont know why, it will naturally get the largest contour and find the points.
Furthermore I was attempting to dynamically find good thresholds for the autocanny and my calculations where not correct.
I changed them to 50 & 150 and they are working fine now.

'''


def main():
    '''
    driver function
    '''
    img2_size = (600,400)
    #bottom left, top left, top right, bottom right
    im2_pts = np.array([[0, 0], [399, 0], [399, 599], [0, 599]])

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
        except (UnidentifiedImageError, OSError):
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

    coords = []
    numOfPoints = 4
    cv2.imshow("Select Points", image)

    # Callback with "standard cartesian" fix:
    cv2.setMouseCallback("Select Points", onClick, (image, coords))

    while len(coords) < numOfPoints:
        cv2.waitKey(1)

    # automatically reorder so no matter the click order, the result is consistent
    pts = np.array(coords, dtype="float32")
    orderedPts = order_points_cartesian(pts)
    
    # If your teacher wants the bottom-left as (0,0), do y_inversion inside onClick!
    cv2.destroyAllWindows()
    return orderedPts

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
        new_y = image.shape[0] - y
        coordsList.append([x, new_y])  # Store coordinates
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Draw a dot
        cv2.imshow("Select Points", image)  # Refresh the image display

def order_points_cartesian(pts):
    """
    Given an array of four (x,y) points in arbitrary order, where (0,0)
    is bottom-left and y increases upward, return them in the order:
      [bottom-left, bottom-right, top-right, top-left].
    """
    rect = np.zeros((4, 2), dtype="float32")

    # sums and differences
    s = pts[:, 0] + pts[:, 1]      # x + y
    d = pts[:, 1] - pts[:, 0]      # y - x

    # bottom-left  => smallest sum
    rect[0] = pts[np.argmin(s)]
    # top-right    => largest sum
    rect[2] = pts[np.argmax(s)]
    
    # bottom-right => smallest difference (y - x)
    rect[1] = pts[np.argmin(d)]
    # top-left     => largest difference
    rect[3] = pts[np.argmax(d)]

    return rect

def auto_canny(image):
    """
    Applies the Canny edge detector using automatically computed threshold values.
    """
    lower = 50
    upper = 150
    edged = cv2.Canny(image, lower, upper)
    return edged

def automaticPoints(image):
    '''
    Performs improved edge detection on the given image to try and find the document.
    Uses polygon approximation (and k-means fallback) to obtain 4 points, and then computes
    a perspective transform to produce a warped (top-down) view of the document.
    '''
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 5)
    
    # Apply auto Canny edge detection with a more typical sigma value
    edges = auto_canny(blurred)
    cv2.imshow('Edges: ', edges)

    # Dilate edges to make them thicker
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours in the edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contourAreaThreshold = 10000
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

    # Use k-means to reduce to 4 points if contour is still noisy
    if contouredDocument is None and len(contours) > 0:
        all_points = np.vstack(contours).squeeze()
        # Apply k-means clustering to get 4 corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
        _, _, centers = cv2.kmeans(np.float32(all_points), 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        contouredDocument = centers.astype(int).reshape((-1,1,2))

    if contouredDocument is None:
        print('No Document Found')
        return None

    # ensure pts are the same shape as the homography points
    pts = contouredDocument.reshape((4, 2)).astype("float32")
    
    # Compute dimensions for the warped image
    widthA = np.linalg.norm(pts[2] - pts[3])
    widthB = np.linalg.norm(pts[1] - pts[0])
    maxWidth = int(max(widthA, widthB))
    
    heightA = np.linalg.norm(pts[1] - pts[2])
    heightB = np.linalg.norm(pts[0] - pts[3])
    maxHeight = int(max(heightA, heightB))
    
    # Define destination points for the warped image
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