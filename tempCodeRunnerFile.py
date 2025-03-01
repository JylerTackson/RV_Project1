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
    blurred = cv2.GaussianBlur(gray, (7, 7), 5)
    
    # Apply Canny Edge Detection
    edges = auto_canny(blurred, 5)

    # Dilate edges to make them thicker
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours in the edge map
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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

    #Using k means to reduce to 4 points if contour is still noisy
    if contouredDocument is None and len(contours) > 0:
        all_points = np.vstack(contours).squeeze()

        #Apply k means clustering to get 4 corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2. TERM_CRITERIA_MAX_ITER, 10, 1)
        _, _, centers = cv2.kmeans(np.float32(all_points), 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        contouredDocument = centers.astype(int).reshape((-1,1,2))

    if contouredDocument is None:
        print('No Document Found')
        return

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