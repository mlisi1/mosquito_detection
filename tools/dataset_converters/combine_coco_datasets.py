import os
import shutil
import json
from PIL import Image
from datetime import datetime
import yaml
import argparse



def parse_args():
    parser = argparse.ArgumentParser(
        description='Combines two COCO datasets')
    parser.add_argument('data1', help='The root path of the first dataset')
    parser.add_argument('data2', help='The root path of the second dataset')
    parser.add_argument(
        'out',
        type=str,
        help='The output folder of the combined dataset')
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


def combine_datasets(data1_path, data2_path, out_path, subdirectories, contributor, description):

    subdirs = ["val", "train", "test"] if subdirectories == None else subdirectories


    info = {
        "year": int(datetime.now().strftime("%Y")),
        "version": "1.0",
        "description": "" if description == None else description,
        "contributor": "" if contributor == None else contributor,
        "url": "",
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    }

    counter = 0

    for sub in subdirs:

        os.makedirs(os.path.join(out_path, sub), exist_ok=True)

        #Load annotations
        with open(os.path.join(data1_path, sub, 'annotations.json'), 'r') as f:
            annotations1 = json.load(f)

        with open(os.path.join(data2_path, sub, 'annotations.json'), 'r') as f:
            annotations2 = json.load(f)

        #Initialize combined annotations 
        combined_annotations = {}
        combined_annotations["info"] = info
        combined_annotations["images"] = annotations1["images"]
        combined_annotations["categories"] = annotations1["categories"]
        combined_annotations["annotations"] = annotations1["annotations"]
 
        image_files1 = [f["file_name"] for f in annotations1["images"]]
        image_files2 = [f["file_name"] for f in annotations2["images"]]

        #Copy images from the first dataset
        for img in image_files1:

            source = os.path.join(data1_path, sub, img)
            dest = os.path.join(out_path, sub, img)

            shutil.copy(source, dest)

        #Copy images from the second dataset        
        for j, img in enumerate(image_files2):

            source = os.path.join(data2_path, sub, img)
            dest = os.path.join(out_path, sub, img)

            filename, ext = os.path.splitext(os.path.basename(source))
            i = 1

            #Handle name conflicts
            while os.path.exists(dest):
                dest = os.path.join(out_path, sub, f"{filename}_{i}{ext}")
                counter += 1
                i+=1

                annotations2["images"][j]["file_name"] = f"{filename}_{i}{ext}"

            shutil.copy(source, dest)


        #Combine classes
        max_cat_id = max([cat["id"] for cat in annotations1["categories"]])
        cat_names = [cat["name"] for cat in annotations1["categories"]]
        cat_id = max_cat_id + 1

        for c in annotations2["categories"]:

            if c['name'] in cat_names:
                continue

            else:
                cat = {}
                cat["name"] = c["name"]
                cat["id"] = cat_id
                combined_annotations["categories"].append(cat)

            cat_id += 1

        #Combine annotations
        max_id = max([img["id"] for img in annotations1["images"]])
        max_annotations_id = max([ann["id"] for ann in annotations1["annotations"]])

        new_image_id = max_id +1
        new_ann_id = max_annotations_id +1
        
        new_cat_names = [cat["name"] for cat in annotations1["categories"]]
        new_cat_ids = [cat["id"] for cat in annotations1["categories"]]
        
        for i, ann in enumerate(annotations2["annotations"]):

            ann["id"] = new_ann_id
            ann["image_id"] = new_image_id

            new_category_id = new_cat_names.index(annotations2["categories"][ann["category_id"]]["name"])
            new_category_id = new_cat_ids[new_category_id]

            ann["category_id"] = new_category_id

            combined_annotations["annotations"].append(ann)


            img = annotations2["images"][i]
            img["id"] = new_image_id

            combined_annotations["images"].append(img)

            new_image_id += 1
            new_ann_id += 1

        new_result_json_path = os.path.join(out_path, sub, "annotations.json")
        with open(new_result_json_path, 'w') as f:
            json.dump(combined_annotations, f)

    if counter > 0:
        print(f'Renamed {counter} files due to name conflicts')
    else:
        print("Correctly combined datasts")





def main():

    args = parse_args()

    combine_datasets(args.data1, args.data2, args.out, args.sub_dirs, args.contributor, args.description)




if __name__ == '__main__':

    main()