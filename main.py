import requests
from io import BytesIO
import numpy as np
from PIL import Image
import matplotlib as plt
import cv2

#The manual Homography/Warping worked with:
#https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.istockphoto.com%2Fphotos%2Fgirls&psig=AOvVaw0JRkdbvOt5x6VIhDX5XVPC&ust=1740885264404000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCJimj6fz54sDFQAAAAAdAAAAABAE
#https://i.redd.it/bebi64i3kbb31.jpg

'''
Explain the idea behind your method:
The drive function is main() of course. This routes the user to retreive Image where I use requests and PIL to request a URL for an image from the user, validates it, then returns it the pillow image object.
Then in main i transfer the pillow object into the a numpy object for data manipulation using openCV. The user is then ruoted to ruoteUser where they are asked if they wante to use Automatic document detection or Manual point selection.

If the user selects Manual they are asked how many points they want to select (It doesnt work for anything other than 4...) then the user is prompted to select the points and they coordinates are saved.
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
After that The contour being returned was warping and returning an all black image. I was very confused by this and how to fix it however I noticed that the dimensions where dynamic so it was returning dynamic contours on the image.
I could not figure out how to fix this unfortanately 





'''


def main():
    '''
    driver function
    '''
    img2_size = (600,400)
    im2_pts = np.array([[0, 0], [0, 599], [399, 599], [399, 0]])

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

def auto_canny(image, sigma=0.33):
    """
    Applies the Canny edge detector using automatically computed threshold values.
    """
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def automaticPoints(image):
    '''
    Performs improved edge detection on the given image to try and find the document.
    Uses polygon approximation (and k-means fallback) to obtain 4 points, and then computes
    a perspective transform to produce a warped (top-down) view of the document.
    '''
    # greyscale the image
    grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # smooth image using gaussian kernel
    smooth_image = cv2.GaussianBlur(grey_image, (7, 7), 5)

    # applying Canny Edge Detection using openCV
    edges = cv2.Canny(smooth_image, threshold1=50, threshold2=150)

    # need to make edges thicker, more robust edges will make following them easier when it comes to drawing contours. 
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    # using openCV's contour method to draw them
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # need to look through the found contours disregard the ones we know aren't the document by ignoring small contours and smoothing jagged edges using openCV convexing
    document_contour = None
    for contour in contours:
        if cv2.contourArea(contour) < 10000:
            continue
        perimeter = cv2.arcLength(contour, True)
        contour_hull = cv2.convexHull(contour)
        approx = cv2.approxPolyDP(contour_hull, 0.007 * perimeter, True)
        if len(approx) == 4:
            document_contour = approx
            break

    # further cleaning the contours if we are noisy, use K-Means to reduce to 4 points
    if document_contour is None and len(contours) > 0:
        all_points = np.vstack(contours).squeeze()

        # apply K-Means clustering to get 4 corner points
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(np.float32(all_points), 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        document_contour = centers.astype(int).reshape((-1,1,2))

    # drawing the cleaned quadrilateral
    detected = cv2.cvtColor(grey_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(detected, [document_contour], -1, (0, 255, 0), 3)  # Draw in green
    cv2.imshow('detected', detected)

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