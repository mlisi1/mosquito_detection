import os
import shutil
import json
from PIL import Image
from datetime import datetime
import yaml
import argparse



def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert images from YOLO format to coco format')
    parser.add_argument('img_path', help='The root path of images, which contains train, val, test folders and dataset.yaml')
    parser.add_argument(
        'out',
        type=str,
        help='The output folder for the coco formatted dataset')
    parser.add_argument(
        '-c',
        '--contributor',
        type=str,
        help='Contributor name for info field'
    )
    parser.add_argument(
        '-d',
        '--description',
        type=str,
        help='Description of the dataset'
    ) 
    parser.add_argument(
        '-s',
        '--sub-dirs',
        type=str,
        nargs='+',
        help='Name of dataset subdirectories. Defauls is train/, test/ and val/'
    )
    args = parser.parse_args()
    return args


def yolo2coco(work_dir, output_folder, contributor, description, subdirectories):


    subdirs = ["val", "train", "test"] if subdirectories == None else subdirectories

    info = {
        "year": int(datetime.now().strftime("%Y")),
        "version": "1.0",
        "description": "" if description == None else description,
        "contributor": "" if contributor == None else contributor,
        "url": "",
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    }


    #CONVERT CATEGORIES
    with open(os.path.join(work_dir, "dataset.yaml"), 'r') as file:
        data = yaml.safe_load(file)
        classes = data["names"]

    coco_classes = []

    for i, c  in enumerate(classes):

        cat = {}
        cat["id"] = i
        cat["name"] = c

        coco_classes.append(cat)


    for sub in subdirs:


        coco = {
            "images": [],        
            "annotations": [],    
            "categories": coco_classes ,
            "info": info
        }

        os.makedirs(os.path.join(output_folder, sub), exist_ok=True)
        image_files = [f for f in os.listdir(os.path.join(work_dir, sub, 'images/'))]

        image_id = 0
        bbox_id = 0


        for img in image_files:

            #CONVERT IMAGES
            image_dict = {}
            image_path = os.path.join(work_dir, sub, 'images/', img)

            with Image.open(image_path) as i:

                image_dict["width"] = i.width
                image_dict["height"] = i.height

            image_dict["id"] = image_id
            output_filename = os.path.join(output_folder, sub, img)
            # coco_filename = os.path.join(output_folder, img)

            image_dict["file_name"] = img
            
            coco["images"].append(image_dict)



            #CONVERT LABELS
            label_name = img.split('.')[0]+'.txt'
            label_path = os.path.join(work_dir, sub, 'labels/', label_name)    

            with open(label_path, 'r') as f:

                label = f.readlines()

            for bb in label:

                annotation = {}
                clss = int(bb.split(' ')[0])
                rel_x = float(bb.split(' ')[1])
                rel_y = float(bb.split(' ')[2])
                rel_w = float(bb.split(' ')[3])
                rel_h = float(bb.split(' ')[4])

                abs_width = abs(rel_w * image_dict["width"])
                abs_height = abs(rel_h * image_dict["height"])
                abs_center_x = rel_x * image_dict["width"]
                abs_center_y = rel_y * image_dict["height"]

                x_min = abs_center_x - (abs_width / 2)
                y_min = abs_center_y - (abs_height / 2)

                annotation["id"] = bbox_id
                annotation["image_id"] = image_id
                annotation["category_id"] = clss
                annotation["iscrowd"] = 0
                annotation["segmentation"] = []
                annotation["ignore"] = 0
                annotation["bbox"] = [x_min,y_min,abs_width,abs_height]
                annotation["area"] = abs_width*abs_height

                coco["annotations"].append(annotation)

                bbox_id += 1       

            image_id += 1

            shutil.copy(image_path, output_filename)


        new_result_json_path = os.path.join(output_folder, sub, "annotations.json")
        with open(new_result_json_path, 'w') as f:
            json.dump(coco, f)


def main():

    args = parse_args()

    yolo2coco(args.img_path, args.out, args.contributor, args.description, args.sub_dirs)



if __name__ == '__main__':

    main()
    


    
    