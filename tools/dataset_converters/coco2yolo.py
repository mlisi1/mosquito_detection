import os
import shutil
import json
from PIL import Image
from datetime import datetime
import yaml
import argparse



def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert images from COCO format to YOLO format')
    parser.add_argument('img_path', help='The root path of images, which contains train, val, test folders. COCO annotations are expected to be inside these folders')
    parser.add_argument(
        'out',
        type=str,
        help='The output folder for the YOLO formatted dataset')
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



def coco_to_yolo_bbox(bbox, img_width, img_height):
    x_min, y_min, width, height = bbox
    x_center = x_min + width / 2.0
    y_center = y_min + height / 2.0

    # Normalize the coordinates
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height

    return x_center, y_center, width, height




def coco2yolo(work_dir, output_folder, contributor, description, subdirectories):


    subdirs = ["val", "train", "test"] if subdirectories == None else subdirectories

    for i in range(len(subdirs)):

        with open(os.path.join(work_dir, subdirs[i], 'annotations.json'), 'r') as f:
            coco_ann = json.load(f)


        os.makedirs(os.path.join(output_folder, subdirs[i], 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, subdirs[i], 'labels'), exist_ok=True)

        images = {img["id"]: img for img in coco_ann["images"]}
        annotations = coco_ann["annotations"]
        categories = {cat["id"]: cat["name"] for cat in coco_ann["categories"]}


        dataset_yaml_path = os.path.join(output_folder, "dataset.yaml")
        with open(dataset_yaml_path, 'w') as yaml_file:
            yaml_file.write(f"train: {os.path.join('train', 'images')}\n")
            yaml_file.write(f"val: {os.path.join('val', 'images')}\n")
            yaml_file.write(f"test: {os.path.join('test', 'images')}\n")

            yaml_file.write("names:\n")
            for category_id, category_name in sorted(categories.items()):
                yaml_file.write(f"  - {category_name}\n")

            


        for img_id, img_data in images.items():

            width = img_data["width"]
            height = img_data["height"]
            source_path = os.path.join(work_dir, subdirs[i], img_data["file_name"])
            dest_path = os.path.join(output_folder, subdirs[i], 'images', img_data["file_name"])
            shutil.copy(source_path, dest_path)


            # Write annotations for the current image
            yolo_annotations = ""
            for ann in annotations:
                if ann["image_id"] == img_id:
                    category_id = ann["category_id"]
                    bbox = ann["bbox"]

                    # Convert bbox to YOLO format
                    yolo_bbox = coco_to_yolo_bbox(bbox, width, height)

                    # Write to YOLO file: <class_id> <x_center> <y_center> <width> <height>
                    yolo_annotations += f"{category_id} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n"

            img_name = img_data["file_name"].split('.')[0]
            ann_path = os.path.join(output_folder, subdirs[i], 'labels', img_name + '.txt')
            with open(ann_path, 'w') as f:
                f.write(yolo_annotations)












def main():

    args = parse_args()

    coco2yolo(args.img_path, args.out, args.contributor, args.description, args.sub_dirs)



if __name__ == '__main__':

    main()
    
