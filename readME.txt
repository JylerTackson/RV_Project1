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