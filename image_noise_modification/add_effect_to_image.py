import cloud_shadow_effect_adder as shadow
import haze_effect_adder as haze
import rain_effect_adder as rain
import cv2

image = cv2.imread("/home/herambjoshi/Downloads/Town02.jpg") # path to your image

shady_image = shadow.add_shadow(image=image, degree_of_shade=0.5)
rainy_image = rain.add_rain(image=image, intensity=1500)
hazy_image = haze.add_fog_random(image=image, reality = 150)

cv2.imwrite("shady.jpg", shady_image)
cv2.imwrite("rainy.jpg", rainy_image)
cv2.imwrite("hazy.jpg", hazy_image)