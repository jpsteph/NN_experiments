import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
import random
import shutil

from PIL import Image

def create_yolo_yaml():
    yaml_str = "train: ../train/images\nval: ../valid/images\ntest: ../test/images\nnc: 2\nnames: ['capacitor', 'resistor']\n"
    yaml = open(r"C:\Users\LENOVO\Documents\_Python Projects\yolo_test\new_data_set\data.yaml", "w")    
    yaml.writelines(yaml_str)
    yaml.close()

def approximate_rectangle(x_lst, y_lst):
    
    # Find the minimum and maximum x and y values
    xmin = min(x_lst)
    xmax = max(x_lst)
    ymin = min(y_lst)
    ymax = max(y_lst)

    # Calculate the center, width, and height
    xcenter = (xmin + xmax) / 2
    ycenter = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin

    return xcenter, ycenter, width, height

def extract_bounding_boxes(file_path):
    bounding_boxes = []
    
    # Open the text file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into components and convert to float
            parts = line.strip().split()

            if len(parts) ==5:
                class_id = int(parts[0])  # The first element is class_id (usually an integer)
                x_center = float(parts[1])  # x center of bounding box
                y_center = float(parts[2])  # y center of bounding box
                width = float(parts[3])  # width of bounding box
                height = float(parts[4])  # height of bounding box
                
                # Store the bounding box information as a tuple or dictionary
                bounding_boxes.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
            elif len(parts)==9:
               x_lst = []
               y_lst = []
               x_lst.append(float(parts[1]))
               x_lst.append(float(parts[3]))
               x_lst.append(float(parts[5]))
               x_lst.append(float(parts[7]))
                
               y_lst.append(float(parts[2]))
               y_lst.append(float(parts[4]))
               y_lst.append(float(parts[6]))
               y_lst.append(float(parts[8]))
               x_center, y_center, width, height = approximate_rectangle(x_lst, y_lst)
    
                # Store the bounding box information as a tuple or dictionary
               bounding_boxes.append({
                    'class_id': int(parts[0]),
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
               })
    return bounding_boxes

# Function to plot image with bounding boxes
def plot_image_with_bboxes(img, bounding_boxes):
    img_height, img_width, _ = img.shape

    # Create the plot
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Loop through bounding boxes and draw them on the image
    for bbox in bounding_boxes:
        # Convert normalized coordinates to pixel values
        x_center = bbox['x_center'] * img_width
        y_center = bbox['y_center'] * img_height
        width = bbox['width'] * img_width
        height = bbox['height'] * img_height

        # Calculate the top-left corner of the bounding box
        x_min = x_center - width / 2
        y_min = y_center - height / 2

        # Create a Rectangle patch and add it to the plot
        rect = patches.Rectangle(
            (x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

    # Show the plot with bounding boxes
    plt.axis('off')  # Hide axis for a cleaner image view
    plt.show()

# Function to flip image and bounding boxes horizontally
def flip_image_and_bboxes(image, bounding_boxes, h_or_v=True):
    
    # Flip the image horizontally using numpy
    if h_or_v:
        flipped_image = np.fliplr(image)
    else:
        flipped_image = np.flipud(image)

    if h_or_v:
        for bbox in bounding_boxes:
            # Calculate the new x_center after flipping
            bbox['x_center'] = 1 - bbox['x_center']  # Invert the x_center (1 - x_center)
    else:
        for bbox in bounding_boxes:
            # Calculate the new x_center after flipping
            bbox['y_center'] = 1 - bbox['y_center']  # Invert the x_center (1 - x_center)
            bbox['x_center'] = bbox['x_center']  # Invert the x_center (1 - x_center)
            

    return flipped_image, bounding_boxes


def rotate_image_and_bboxes(image, bounding_boxes, angle):

    if angle == 180:
        image, bounding_boxes = flip_image_and_bboxes(image, bounding_boxes, h_or_v=True)
        image, bounding_boxes = flip_image_and_bboxes(image, bounding_boxes, h_or_v=False)
        for bbox in bounding_boxes:
            bbox['x_center'] = bbox['x_center']
        return image, bounding_boxes

    # Get the image dimensions
    img_height, img_width, _ = image.shape

    if angle==90:
        rotated_image = np.rot90(image, k=1)  # Rotate by 90 degrees
    elif angle==270:
        rotated_image = np.rot90(image, k=3)
    
    img_height, img_width, _ = rotated_image.shape
   
    # Adjust bounding box coordinates based on rotation angle
    rotated_bboxes = []
    for bbox in bounding_boxes:
        # Convert normalized bounding box coordinates to pixel values
        x_center = bbox['x_center'] 
        y_center = bbox['y_center'] 
        width = bbox['width'] 
        height = bbox['height'] 
        
        x_min = x_center - width/2
        x_max = x_center + width/2
        y_min = y_center - height/2
        y_max = y_center + height/2

        if angle == 90:
            new_xmin = y_min
            new_ymin = (img_width-x_max*img_width)/img_width
            new_xmax = y_max
            new_ymax = (img_width-x_min*img_width)/img_width
        elif angle == 270:
            new_ymin = x_min
            new_xmin = (img_width-y_max*img_width)/img_width
            new_ymax = x_max
            new_xmax = (img_width-y_min*img_width)/img_width
                    

        new_width = new_xmax-new_xmin
        new_height = new_ymax-new_ymin

        new_x_center = new_xmin + new_width/2
        new_y_center = new_ymin + new_height/2
        # Normalize back to the new image dimensions
        if angle==90:
            rotated_bboxes.append({
            'class_id': bbox['class_id'],
            'x_center': new_x_center,
            'y_center': new_y_center,
            'width': new_width,
            'height': new_height
            })
        elif angle==270:
                        rotated_bboxes.append({
            'class_id': bbox['class_id'],
            'x_center': new_x_center,
            'y_center': new_y_center,
            'width': new_width,
            'height': new_height
            })

    return rotated_image, rotated_bboxes

def zoom_into_image_and_bboxes(image, bounding_boxes, zoom_percentage):
    # Get the image dimensions
    img_height, img_width, _ = image.shape
    
    # Calculate the new dimensions after zooming
    zoom_factor = 1 - zoom_percentage
    new_width_image = int(img_width * zoom_factor)
    new_height_image = int(img_height * zoom_factor)
    
    # Calculate the cropping box to zoom into the center of the image
    left = (img_width - new_width_image) // 2
    top = (img_height - new_height_image) // 2
    right = left + new_width_image
    bottom = top + new_height_image
    
    # Crop the image to zoom into the center
    zoomed_image = image[top:bottom, left:right]
    
    # Adjust bounding box coordinates for zoom
    zoomed_bboxes = []
    for bbox in bounding_boxes:
        # Convert normalized bounding box coordinates to pixel values
        x_center = bbox['x_center'] * img_width
        y_center = bbox['y_center'] * img_height
        width = bbox['width'] * img_width
        height = bbox['height'] * img_height
        
        # Adjust the bounding box coordinates based on the zoom
        # The new x_center and y_center are mapped to the zoomed image dimensions
        new_x_center = (x_center - left) / new_width_image
        new_y_center = (y_center - top) / new_height_image
        new_width = width / new_width_image
        new_height = height / new_height_image

        if (new_x_center + new_width/2 < 0) and (new_x_center - new_width/2 > new_width_image) and (new_y_center + new_height/2 < 0) and (new_y_center - new_height/2 > new_height_image):
            pass
        elif new_y_center < 0 or new_x_center < 0:
            pass
        else:
            zoomed_bboxes.append({
                'class_id': bbox['class_id'],
                'x_center': new_x_center,
                'y_center': new_y_center,
                'width': new_width,
                'height': new_height
            })

    return zoomed_image, zoomed_bboxes


def get_files_from_folder(folder_path):
    global all_files

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            all_files.append(os.path.join(root, file))


if __name__=="__main__":

    all_files = []
    debug = False

    folder_path_1 = r'C:\Users\LENOVO\Documents\_Python Projects\yolo_test\Everything.v8i.yolov11\test\images'
    folder_path_2 = r'C:\Users\LENOVO\Documents\_Python Projects\yolo_test\Everything.v8i.yolov11\test\labels'
    folder_path_3 = r'C:\Users\LENOVO\Documents\_Python Projects\yolo_test\Everything.v8i.yolov11\train\images'
    folder_path_4 = r'C:\Users\LENOVO\Documents\_Python Projects\yolo_test\Everything.v8i.yolov11\train\labels'
    folder_path_5 = r'C:\Users\LENOVO\Documents\_Python Projects\yolo_test\Everything.v8i.yolov11\valid\images'
    folder_path_6 = r'C:\Users\LENOVO\Documents\_Python Projects\yolo_test\Everything.v8i.yolov11\valid\labels'

    
    #img = mpimg.imread(r'C:\Users\LENOVO\Documents\_Python Projects\yolo_test\Everything.v6i.yolov11\train\images\pcb37_jpg.rf.d63e2bc6c3f12a05b512e713ba944e3b.jpg')
    #box = extract_bounding_boxes(r'C:\Users\LENOVO\Documents\_Python Projects\yolo_test\Everything.v6i.yolov11\train\labels\pcb37_jpg.rf.d63e2bc6c3f12a05b512e713ba944e3b.txt')
    #plot_image_with_bboxes(img, box)
    #plot_image_with_bboxes(mpimg.imread(r'C:\Users\LENOVO\Documents\_Python Projects\yolo_test\new_data_set\train\images\64.jpg'), extract_bounding_boxes(r'C:\Users\LENOVO\Documents\_Python Projects\yolo_test\new_data_set\train\labels\64.txt'))

    # Get files from each folder
    get_files_from_folder(folder_path_1)
    get_files_from_folder(folder_path_2)

    get_files_from_folder(folder_path_3)
    get_files_from_folder(folder_path_4)
    get_files_from_folder(folder_path_5)
    get_files_from_folder(folder_path_6)

    
    #sort by everything except .jpg or .txt
    all_files.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])

    # Print the sorted list of files
    data_lst = []
    i=0
    file_pair=[]
    for file in all_files:
        file_pair.append(file)

        i+=1
        if i==2:
            i=0
            file_pair=[]
            data_lst.append(file_pair)

    random.shuffle(data_lst)

    directory_path = r"C:\Users\LENOVO\Documents\_Python Projects\yolo_test\new_data_set"
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
        print(f"Directory '{directory_path}' and all its contents have been deleted.")

    os.makedirs(directory_path, exist_ok=True)  

    subfolders = ["train", "test", "valid"]

    # Create each subfolder inside the main directory
    for subfolder in subfolders:
        subfolder_path = os.path.join(directory_path, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)
        print(f"Subfolder '{subfolder_path}' created successfully!")

        # Create two new subfolders inside each folder
        inner_subfolders = ["images", "labels"]
        for inner_subfolder in inner_subfolders:
            inner_folder_path = os.path.join(subfolder_path, inner_subfolder)
            os.makedirs(inner_folder_path, exist_ok=True)
            print(f"Inner subfolder '{inner_folder_path}' created successfully!")

    create_yolo_yaml()

    num_dataset = len(data_lst)
    print('Images: ' + str(num_dataset))

    scale_num = 2

    train = int(num_dataset*0.6*scale_num)
    test = int(num_dataset*0.05*scale_num)
    valid = int(num_dataset*0.35*scale_num)

    if (train+test+valid) < int(num_dataset*scale_num):
        while (train+test+valid) < int(num_dataset*scale_num):
            test+=1

    if (train+test+valid) > int(num_dataset*scale_num):
        while (train+test+valid) > int(num_dataset*scale_num):
            test-=1
    

    print('Num Train Images: ' + str(train))
    print('Num Test Images: ' + str(test))
    print('Num Valid Images: ' + str(valid))

    data_lst = [item for item in data_lst if item != []]

    count = 0
    
    for fp in data_lst:
        

        #rand_val = random.uniform(0, 0.2)        
        #img, bounding_boxes = zoom_into_image_and_bboxes(img, bounding_boxes, rand_val)
        
        scale_count = scale_num
        while(scale_count>0):
            
            img = mpimg.imread(fp[0])
            bounding_boxes = extract_bounding_boxes(fp[1])

            random_bool_trans = random.choice([True, False])
            if random_bool_trans:
                random_bool = random.choice([True, False])
                if random_bool:
                    print('FLIPH')
                else:
                    print('FLIPV')
                img_n, bounding_boxes_n = flip_image_and_bboxes(img, bounding_boxes, h_or_v=random_bool)
            else:
                rot = random.choice([90, 180, 270])
                print('ROT '+ str(rot))
                img_n, bounding_boxes_n = rotate_image_and_bboxes(img, bounding_boxes, angle=rot)


            set_choice = random.randint(1,3)

            if train==0:
                set_choice = random.randint(2,3)
            if test==0:
                set_choice = random.choice([1,3])
            if valid==0:
                set_choice = random.randint(1,2)

            if train==0 and test==0:
                set_choice=3
            elif test==0 and valid==0:
                set_choice=1
            elif train==0 and valid==0:
                set_choice=2   

            path_str = ''        
            if set_choice==1:
                path_str = 'train'
                train-=1
            elif set_choice==2:
                path_str = 'test'
                test-=1
            elif set_choice==3:
                path_str = 'valid'
                valid-=1
                
            if path_str=='':
                raise ValueError

            img_path = os.path.join(directory_path, path_str+'/images')
            img_path = os.path.join(img_path, str(count)+'_'+str(scale_count)+'.jpg')
            image_to_save = Image.fromarray(img_n)
            image_to_save.save(img_path)        

            box_path = os.path.join(directory_path, path_str+'/labels/'+str(count)+'_'+str(scale_count)+'.txt')
            box_txt = open(box_path, "w")

            for b in bounding_boxes_n:
                write_str = str(round(b['class_id'],5)) +' '+str(round(b['x_center'],5))+' '+str(round(b['y_center'],5))+' '+str(round(b['width'],5))+' '+str(round(b['height'],5))+'\n'
                box_txt.writelines(write_str)
            box_txt.close()

            if debug:
                img_n = mpimg.imread(img_path)
                bounding_boxes_n = extract_bounding_boxes(box_path)
                plot_image_with_bboxes(img_n, bounding_boxes_n)
            

            print(count)
            count+=1
            scale_count-=1

    print(train)
    print(test)
    print(valid)



