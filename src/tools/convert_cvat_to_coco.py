import sys
import json
import os

def convert_cvat_to_coco(input_json_file, output_json_file):
    cvat_coco = json.load(open(input_json_file, "r"))

    # Need to correct the path of images
    images = cvat_coco["images"]
    image_dir = os.path.join(os.path.dirname(input_json_file), \
            os.path.splitext(os.path.basename(input_json_file))[0])
    image_rel_path = os.path.relpath(image_dir, os.path.dirname(input_json_file))

    for image_obj in images:
        file_name = image_obj["file_name"]
        if os.path.exists(os.path.join(image_dir, file_name)):
            file_name = os.path.join(image_rel_path, file_name)
        elif os.path.exists(os.path.join(image_dir, file_name+".jpg")):
            file_name = os.path.join(image_rel_path, file_name+".jpg")
        else:
            print("%s Not Found" % file_name)
        image_obj["file_name"] = file_name

    json.dump(cvat_coco, open(output_json_file, "w"))
    print("Saved to", output_json_file)

def main():
    if len(sys.argv)<2:
        print("Usage: python %s [coco_json_file]" % sys.argv[0])
        print("Assume images is in [coco_json_file_name] folder")
        return
    cvat_coco_json_file = sys.argv[1]
    correct_coco_json_output_file = os.path.join( \
                                     os.path.dirname(cvat_coco_json_file), \
                                     os.path.splitext(os.path.basename(cvat_coco_json_file))[0] +
                                     "_correct_output.json")

    convert_cvat_to_coco(cvat_coco_json_file, correct_coco_json_output_file)


if __name__ == "__main__":
    main()