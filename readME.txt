Goal:
    -Make a python program that rectifies images from documents (like camscanner).
    
Deliver:
    1. Code with comments. Program must run in colab.
    2. Report. Description of the method used with conclusions.

General Steps:
  There are 3 steps to do this:
    1. Find the corners of the document to be rectified
    2. Find the transformation that maps points A, B, C, D to a new set of points in (0,0), (H,0), (H,W),and (0,W). (see figure below)
    3. Perform warping operation (Check slide with recipe) using the found transformation.

Manual mode (80%)
    Your first task is to probe your algorithm works. Try your algorithm in two pictures from your choice.
      - Make sure to use the highest resolution possible, and the image looks like a projection.
      - Avoid fisheye lenses or lenses with significant barrel distotion (do straight lines come out straight?).
        a. You will provide the user a way to select the four corner points (it can be typing, but let to know the grader some coordinates to test).
        b. Use the method you did on homework 5, H = computeH(im1_pts,im2_pts) to compute the homography that take you from input image to the transformed image.
        c. Perform backward warping with the method you created in homework 5. imwarped =warpImage(im,H).
           You can use the opencv method to perform warping in case that your method does not work or want to compare it.
        d. Show your warped image.

Using OpenCV's 'cv2.setMouseCallback("Image", click_event, (coords, img))':
  -This function allowed me to call a click_event function from within another function.
   Furthermore, within the paramters I was able to pass a list where I reserached about how python handles mutable vs immutable data types
   I passed a list to a function and eddited it and since the list is mutable the edits effect the original object.
  -Passing information to and from the onclick and manualPoints function using this was very easy due to the way Python handles mutuable objects
   therefore updating the image with the selected point was also very easy.
  -To call the onClick function for the specific number of clicks the user wanted do was tedious to figure out.
   You have to call the .setMouseCallback to setup the callback feature of cv2 then use .waitKey to keep the window responsive until your condition is met.

Using PIL image objects with OpenCV image Objects:
  -To retrieve the object and return it to main I used: 
                      return(Image.open(BytesIO(response.content)))
  however this returned a PIL image object which is not compatible with openCV functions as they expect to receive an image in the form of a NumPy image array.
  To fix this, simply just transfer your image into a numpy object after retreiving your iamge.


