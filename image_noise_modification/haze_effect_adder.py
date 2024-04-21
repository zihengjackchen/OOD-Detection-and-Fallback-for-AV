import cv2
import numpy as np

def add_blur(image, x,y,hw):    
    image[y:y+hw, x:x+hw,1] = image[y:y+hw, x:x+hw,1]+1    
    image[:,:,1][image[:,:,1]>255] = 255 
	##Sets all values above 255 to 255
    image[y:y+hw, x:x+hw,1] = cv2.blur(image[y:y+hw, x:x+hw,1] ,(10,10))    
    return image
          

def generate_random_blur_coordinates(imshape, reality, hw):    
    blur_points=[]    
    midx= imshape[1]//2-hw-100    
    midy= imshape[0]//2-hw-100    
    index=1    
    count = 0
    while(midx>-100 or midy>-100): 
		## radially generating coordinates        
        for i in range(reality*index):            
            x= np.random.randint(midx,imshape[1]-midx-hw)            
            y= np.random.randint(midy,imshape[0]-midy-hw)
            if ((x < 0 or y < 0) or (x >= imshape[1] - hw or y >= imshape[0] - hw)):
                count += 1
            else:         
                blur_points.append((x,y))        
        midx-=250*imshape[1]//sum(imshape)        
        midy-=250*imshape[0]//sum(imshape)        
        index+=1    
    return blur_points    

def add_fog(image, haze_list):    
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) 
    ## Conversion to HLS      
    hw=100    
    image_HLS[:,:,1]=image_HLS[:,:,1]*0.8    
    for haze_points in haze_list:         
        image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 
        ##Sets all values above 255 to 255
        image_HLS = add_blur(image_HLS, haze_points[0],haze_points[1], hw) 
        ## adding all shadow polygons on empty mask, single 255 denotes only red channel    
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB    
        
    return image_RGB

def add_fog_random(image, reality = 100):
    image = image.copy()
    image_RGB = cv2.cvtColor(image,cv2.COLOR_BGRA2RGB) 
    image_HLS = cv2.cvtColor(image_RGB,cv2.COLOR_RGB2HLS) 
    ## Conversion to HLS      
    hw=100    
    image_HLS[:,:,1]=image_HLS[:,:,1]*0.8
    haze_list = generate_random_blur_coordinates(image.shape, reality, 100)
    for haze_points in haze_list:         
        image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 
        ##Sets all values above 255 to 255
        image_HLS = add_blur(image_HLS, haze_points[0],haze_points[1], hw) 
        ## adding all shadow polygons on empty mask, single 255 denotes only red channel    
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB    
    image_BGRA = cv2.cvtColor(image_RGB,cv2.COLOR_RGB2BGRA) ## Conversion to RGB 
    return image_BGRA

# Create a VideoCapture object and read from input file 
#-------------- TODO ------------- please provide your video/image path here
if (__name__ == "main"):
    cap = cv2.VideoCapture('/home/herambjoshi/Desktop/video.mp4') 
    haze_list = None
    # Check if camera opened successfully 
    if (cap.isOpened()== False): 
        print("Error opening video file") 

    # Read until video is completed 
    while(cap.isOpened()): 
        
    # Capture frame-by-frame    
        ret, frame = cap.read() 
        if ret == True: 
        # Display the resulting frame 
            if (haze_list is None):
                haze_list = generate_random_blur_coordinates(frame.shape, 50,100)
                print(len(haze_list))
            cv2.imshow('Frame', add_fog(frame, haze_list=haze_list)) 
            print("new frame")
            
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
