import cv2
import numpy as np

def generate_random_lines(imshape,slant,drop_length, drops_per_frame):    
    drops=[]    
    for i in range(drops_per_frame): 
        if slant<0:            
            x= np.random.randint(slant,imshape[1])        
        else:            
            x= np.random.randint(0,imshape[1]-slant)
        y= np.random.randint(0,imshape[0]-drop_length)
        drops.append((x,y))    
    return drops            

def add_rain(image, intensity = 1500):  
    image = image.copy()      
    imshape = image.shape
    slant_extreme=10    
    slant= np.random.randint(-slant_extreme,slant_extreme)     
    drop_length=20    
    drop_width=2    
    drop_color=(200,200,200) ## a shade of gray ## reduce from 200 if you want grayer rain. 
    drop_range = 20  ## increase to add random colored drops
    rain_drops= generate_random_lines(imshape,slant,drop_length, intensity) # set intensity        
    for rain_drop in rain_drops:
        x = np.random.uniform() * drop_range
        drop_color = (drop_color[0] + x, drop_color[1] + x, drop_color[2] + x)       
        cv2.line(image,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)    
    image= cv2.blur(image,(7,7)) ## rainy view are blurry        
    brightness_coefficient = 0.7 ## rainy days are usually shady     
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS    
    image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)    
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB    
    return image_RGB

# Create a VideoCapture object and read from input file 
#-------------- TODO ------------- please provide your video/image path here

if (__name__ == "main"):
    cap = cv2.VideoCapture('/home/herambjoshi/Desktop/video.mp4') 

    # Check if camera opened successfully 
    if (cap.isOpened()== False): 
        print("Error opening video file") 

    # Read until video is completed 
    while(cap.isOpened()): 
        
    # Capture frame-by-frame 
        ret, frame = cap.read() 
        if ret == True: 
        # Display the resulting frame 
            cv2.imshow('Frame', add_rain(frame)) 
            
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
