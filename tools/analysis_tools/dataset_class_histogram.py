import json
import argparse
import matplotlib.pyplot as plt
import os




def parse_args():
    parser = argparse.ArgumentParser(
        description='Creates the class histogram from a COCO formatted dataset')
    parser.add_argument('dataset_path', help='The root path of the dataset, which contains train, val, test folders. COCO annotations are expected to be inside these folders')
    parser.add_argument(
        '-o',
        '--out',
        type=str,
        help='If specified, saves the histogram to that path'
    )
    parser.add_argument(
        '-p',
        '--phase',
        type=str,
        choices=['train', 'test', 'val', 'all'],
        default='all',
        help='Which dataset section to use'
    ) 
    parser.add_argument(
        '-s',
        '--show',
        help='Shows the plot',
        action='store_true'
    )
    args = parser.parse_args()
    return args



def count_classes(annotations):

    classes = annotations["categories"]

    counter = {}

    for c in classes:

        counter[c["id"]] = 0

    for ann in annotations["annotations"]:

        counter[ann["category_id"]] += 1

    return counter





def hist(dataset_path, out, phase, show):


    subpath = [phase] if phase != "all" else ["train", "test", "val"]

    classes = None


    for sub in subpath:

        ann_path = os.path.join(dataset_path, sub, 'annotations.json')

        with open(ann_path, 'r') as f:
            annotations = json.load(f)


        if classes == None:

            classes = count_classes(annotations)

        else:

            counter = count_classes(annotations)

            for id_ in classes.keys():

                classes[id_] += counter[id_]

    bins = list(classes.keys())
    frequencies = list(classes.values())

    plt.bar(bins, frequencies, width=0.9, color='orange', edgecolor='black', alpha=0.7)

    for bin, freq in zip(bins, frequencies):
        label = annotations["categories"][bin]["name"]
        plt.text(bin, -2, label, ha='center', va='top', fontsize=12)

    plt.title(f'Dataset classes distribution - Phase: {phase}')
    plt.ylabel('Instances')
    plt.xticks([])

    if show:
        plt.show()

    if out != None:

        if os.path.isfile(out):

            plt.savefig(out)

        else:

            plt.savefig(os.path.join(out, 'classes_distribution.png'))
    


        









def main():

    args = parse_args()

    hist(args.dataset_path, args.out, args.phase, args.show)



if __name__ == '__main__':

    main()