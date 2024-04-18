from PIL import Image
import os
import csv
import argparse

def get_image_pixels(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        pixels = [(x, y, *img.getpixel((x, y))) for x in range(width) for y in range(height)]
        return width, height, pixels

def save_pixels_to_individual_csv(image_folder, csv_folder):
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
        
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            width, height, pixels = get_image_pixels(image_path)
            
            csv_filename = os.path.splitext(filename)[0] + '.csv'
            csv_path = os.path.join(csv_folder, csv_filename)
            
            with open(csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['X', 'Y', 'R', 'G', 'B', 'A'])
                
                for x, y, r, g, b, a in pixels:
                    writer.writerow([x, y, r, g, b, a])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract pixel data from images.')
    parser.add_argument('-left', action='store_true', help='Process images in left folder')
    parser.add_argument('-middle', action='store_true', help='Process images in middle folder')
    parser.add_argument('-right', action='store_true', help='Process images in right folder')
    
    args = parser.parse_args()
    
    if args.left:
        image_folder = '/mnt/shared/home/weihang6/OOD-Detection/OOD-Detection/left_images'
        csv_folder = '/mnt/shared/home/weihang6/OOD-Detection/Distribution/left'
    elif args.middle:
        image_folder = '/mnt/shared/home/weihang6/OOD-Detection/OOD-Detection/middle_images'
        csv_folder = '/mnt/shared/home/weihang6/OOD-Detection/Distribution/middle'
    elif args.right:
        image_folder = '/mnt/shared/home/weihang6/OOD-Detection/OOD-Detection/right_images'
        csv_folder = '/mnt/shared/home/weihang6/OOD-Detection/Distribution/right'
    else:
        print("Please specify a folder using -left, -middle, or -right.")
        exit(1)
        
    save_pixels_to_individual_csv(image_folder, csv_folder)
