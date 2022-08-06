'''
    convert coco file to yolov5 format
    vectormaster
    2022/08/05
'''

import json
import os
from os import path
from functools import reduce
import glob
import shutil

INPUT_DIR = 'data/XRAY/right'
OUTPUT_DIR = 'output'

def convert_to_min_max_box(bbox_coco):              # (xmin, ymin, width, height) => (xmin, ymin, xmax, ymax)
    return {
        'xmin': bbox_coco[0],
        'ymin': bbox_coco[1],
        'xmax': bbox_coco[0] + bbox_coco[2],
        'ymax': bbox_coco[1] + bbox_coco[3]
    }

def get_big_box(n_boxes):                           # get bounding box of 2 or more boxes
    return {
        'xmin': min(map(lambda box: box['xmin'], n_boxes)),
        'ymin': min(map(lambda box: box['ymin'], n_boxes)),
        'xmax': max(map(lambda box: box['xmax'], n_boxes)),
        'ymax': max(map(lambda box: box['ymax'], n_boxes)),
    }

def group_teeth_ids(ids):                           # group teeth (ids) into 2 or 3 sets
    ids.sort()
    groups = []
    for (i, teeth_id) in enumerate(ids):
        n = 1
        group = [teeth_id]
        for j in range(i + 1, len(ids)):
            if n >= 3 or teeth_id // 10 != ids[j] // 10:
                break
            group.append(ids[j])
            n += 1
            groups.append(group.copy())
    return groups

def convert_yolov5_str(class_id, bbox, image_width, image_height):          # coco format => yolov5 format  (for one box)
    bbox_center_x = round((bbox['xmin'] + bbox['xmax']) / (2 * image_width), 6) 
    bbox_center_y = round((bbox['ymin'] + bbox['ymax']) / (2 * image_height), 6)
    bbox_width = round((bbox['xmax'] - bbox['xmin']) / image_width, 6) 
    bbox_height = round((bbox['ymax'] - bbox['ymin']) / image_height, 6)

    return "{} {} {} {} {}".format(class_id, bbox_center_x, bbox_center_y, bbox_width, bbox_height)

def convert_group_yolov5(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):       # main function to convert all images in the input directory into yolov5 format
    json_files = glob.glob(path.join(input_dir, "*.json"))
    labels = []
    for json_file in json_files:
        print("Parsing {}".format(json_file))
        f = open(json_file)
        raw_data = json.load(f)
        f.close()
        category_id_to_name_dict = {}   # category_id => tooth_id
        bboxes = {}                     # tooth_id => bbox

        image_width, image_height = raw_data['images'][0]['width'], raw_data['images'][0]['height']
        
        for cat in raw_data['categories']:
            tooth_id = cat['name'][6:]
            try:
                tooth_id = int(tooth_id)                # just normal tooth with number
                category_id_to_name_dict[cat['id']] = tooth_id
            except:
                pass

        for annotation in raw_data['annotations']:
            box = convert_to_min_max_box(annotation['bbox'])
            cat_id = annotation['category_id']
            
            if not cat_id in category_id_to_name_dict:      # just normal tooth with number
                continue
            bboxes[category_id_to_name_dict[cat_id]] = box

        try:                            # make output directory if not exits
            os.mkdir(output_dir) 
        except:
            pass

            
        teeth_ids = list(bboxes.keys())
        groups = group_teeth_ids(teeth_ids)
        label_strs = []
        for group in groups:
            group_label = reduce(lambda a,b: str(a) + '-' + str(b), group)          # group label  e.g. 14-15-16, 16-17, ...
            class_id = None

            try:
                class_id = labels.index(group_label)
            except:
                class_id = len(labels)
                labels.append(group_label)

            group_boxes = list(map(lambda teeth_id: bboxes[teeth_id], group))
            big_box = get_big_box(group_boxes)
            label_strs.append(convert_yolov5_str(class_id, big_box, image_width, image_height))     
        
        label_strs_reduce = "" if len(label_strs) == 0 else reduce(lambda a, b: a + '\n' + b, label_strs)           # concat all boxes to fit in one txt file
        txt_file_name = path.basename(json_file)[:-5] + '.txt'
        image_file_name = raw_data['images'][0]['file_name']
        f = open(path.join(output_dir, txt_file_name), "w")                         # write to txt file
        f.write(label_strs_reduce)
        f.close()
        shutil.copyfile(path.join(input_dir, image_file_name), path.join(output_dir, image_file_name))          # copy image file
        print("{} converted".format(json_file))

if __name__ == "__main__":
    convert_group_yolov5(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)