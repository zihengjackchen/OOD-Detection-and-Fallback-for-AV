import cv2
import numpy as np

def add_shadow(image, degree_of_shade):    
    image = image.copy()
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) 
    ## Conversion to HLS    
    image_HLS = np.array(image_HLS, dtype = np.float64)     
    random_brightness_coefficient = degree_of_shade
    ## generates value between 0.5 and 1.5    
    image_HLS[:,:,1] = image_HLS[:,:,1]*random_brightness_coefficient 
    ## scale pixel values up or down for channel 1(Lightness)    
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 
    ##Sets all values above 255 to 255    
    image_HLS = np.array(image_HLS, dtype = np.uint8)    
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB    
    return image_RGB

# Create a VideoCapture object and read from input file 
#-------------- TODO ------------- please provide your video/image path here
if (__name__ == "main"):
    cap = cv2.VideoCapture('/home/herambjoshi/Desktop/video.mp4')  # imread for images

    # Check if camera opened successfully 
    if (cap.isOpened()== False): 
        print("Error opening video file") 

    # Read until video is completed 
    while(cap.isOpened()): 
        
    # Capture frame-by-frame 
        ret, frame = cap.read() 
        if ret == True: 
        # Display the resulting frame 
            cv2.imshow('Frame', add_shadow(frame, 0.3)) 
            
        # Press Q on keyboard to exit 
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break

    # Break the loop 
        else: 
            break

    # When everything done, release 
    # the video capture object 
    cap.release() 

    # Closes all the frames 
    cv2.destroyAllWindows() 
