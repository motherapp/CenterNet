import sys
import glob
import json
import random
import os

def main():
    if len(sys.argv)<2:
        print("Usage: python %s [annotations folder]" % sys.argv[0])
        print("Assume all annotation json has the same class(es)")
        return
    ann_dir = sys.argv[1]

    # load coco data
    coco_data_list = []
    category_dict = {}
    for file_name in glob.glob(ann_dir + "/*.json"):
        if os.path.basename(file_name).find("train.json")>=0 or \
           os.path.basename(file_name).find("val.json")>=0 or \
           os.path.basename(file_name).find("test.json")>=0:
           continue

        coco_data = json.load(open(file_name, "r"))
        coco_data_list.append(coco_data)

        for cat in coco_data["categories"]:
            if cat["name"] in category_dict:
                continue
            category_dict[cat["name"]] = cat["id"]

    # modify image id, category id, annotation id
    offset_image_id = 0
    offset_category_id = 0
    ann_dict = {}
    for coco_data in coco_data_list:
        cat_id_to_id = {}
        for cat in coco_data["categories"]:
            cat_id_to_id[cat["id"]] = category_dict[cat["name"]]
            cat["id"] = cat_id_to_id[cat["id"]]

        max_image_offset_id = offset_image_id
        for image in coco_data["images"]:
            max_image_offset_id = max(max_image_offset_id, image["id"]+offset_image_id + 1)
            image["id"] = image["id"]+offset_image_id

        max_category_offset_id = offset_category_id
        for ann in coco_data["annotations"]:
            max_category_offset_id = max(max_category_offset_id, ann["id"]+offset_category_id + 1)
            ann["id"] = ann["id"]+offset_category_id
            ann["image_id"] = ann["image_id"]+offset_image_id
            ann["category_id"] = cat_id_to_id[ann["category_id"]]
            ann_list = ann_dict.get(ann["image_id"], [])
            ann_list.append(ann)
            ann_dict[ann["image_id"]] = ann_list

        offset_image_id = max_image_offset_id
        offset_category_id = max_category_offset_id

    # merge
    merge_coco_data = {
        "licenses": [{"name": "", "id": 0, "url": ""}], 
        "info": {"contributor": "", "date_created": "2019-11-15", "description": "Lok Ma Chau", "url": "", "version": 3, "year": "2019"},
        "categories": coco_data_list[0]["categories"],
        "images": [],
        "annotations": [] }
    for coco_data in coco_data_list:
        merge_coco_data["images"] = merge_coco_data["images"] + coco_data["images"]        
        merge_coco_data["annotations"] = merge_coco_data["annotations"] + coco_data["annotations"]

    # train, val, test (70%, 15%, 15%)
    random.seed(34278265)
    random.shuffle(merge_coco_data["images"])

    train_offset = int(len(merge_coco_data["images"]) * 0.7)
    validation_offset = int(len(merge_coco_data["images"]) * 0.85)

    train_coco_data = merge_coco_data.copy()
    train_coco_data["images"] = merge_coco_data["images"][:train_offset]
    train_coco_data["annotations"] = []
    for image in train_coco_data["images"]:
        for ann in ann_dict[image["id"]]:
            train_coco_data["annotations"].append(ann)

    validation_coco_data = merge_coco_data.copy()
    validation_coco_data["images"] = merge_coco_data["images"][train_offset:validation_offset]
    validation_coco_data["annotations"] = []
    for image in validation_coco_data["images"]:
        for ann in ann_dict[image["id"]]:
            validation_coco_data["annotations"].append(ann)

    test_coco_data = merge_coco_data.copy()
    test_coco_data["images"] = merge_coco_data["images"][validation_offset:]
    test_coco_data["annotations"] = []
    for image in test_coco_data["images"]:
        for ann in ann_dict[image["id"]]:
            test_coco_data["annotations"].append(ann)

    # save to train.json, val.json and test.json
    json.dump(train_coco_data, open(os.path.join(ann_dir, "train.json"), "w"))
    json.dump(validation_coco_data, open(os.path.join(ann_dir, "val.json"), "w"))
    json.dump(test_coco_data, open(os.path.join(ann_dir, "test.json"), "w"))


    print(len(train_coco_data["images"]), len(validation_coco_data["images"]), len(test_coco_data["images"]))
    

if __name__ == "__main__":
    main()
