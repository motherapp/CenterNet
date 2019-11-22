# Develop

This document provides tutorials to develop CenterNet. `lib/src/opts` lists a few more options that the current version supports.

## 1. New dataset
Basically there are three steps:

- Convert the dataset annotation to [COCO format](http://cocodataset.org/#format-data). Please refer to [src/tools/convert_kitti_to_coco.py](../src/tools/convert_kitti_to_coco.py) for an example to convert kitti format to coco format.
- Create a dataset intilization file in `src/lib/datasets/dataset`. In most cases you can just copy `src/lib/datasets/dataset/coco.py` to your dataset name and change the category information, and annotation path.
- Import your dataset at `src/lib/datasets/dataset_factory`.

### New dataset from CVAT ()
Current folder: [CenterNet Root]/src

~~~
cd src
export PYTHONPATH=
~~~

[dataset]: the name of custom dataset
Folder structure:
- ../data/[dataset]/annotations: coco annotation json (train.json, val.json and test.json)
- ../data/[dataset]/images (dataset images and cvat coco annotation json)
- ../exp/ctdet/[experiment_name] (folder of the training model and logs)

Steps to prepare dataset:
1. Download cvat coco annotation json ([annotation_name].json) into data/[dataset]/images
2. Copy images to data/[dataset]/images/[annotation_name] if the CVAT task is image annotation
3. Extract images from video to data/[dataset]/images/[annotation_name] if the CVAT task is video annotation by using this command:

~~~
python tools/cvat_get_frames_by_video.py [input_video.mp4] ../data/[dataset]/images/[annotation_name] [framestep]
~~~

4. Modify annotation json files ../data/[dataset]/images/*.json to use relative path of images

~~~
python tools/convert_cvat_to_coco.py ../data/[dataset]/images/[annotation_name].json
~~~

5. Copy modified annotation json files to annotation folder

~~~
cp ../data/[dataset]/images/*_correct_output.json ../data/[dataset]/annotations/
~~~

6. Create train, val, test dataset

~~~
python tools/merge_coco_json_split_train_val_test.py ../data/[dataset]/annotations/ ../data/[dataset]/images/
~~~

7. Prepare mean and std dev of images: python tools/get_mean_std_dataset.py

~~~
python tools/get_mean_std_dataset.py ../data/[dataset]/annotations/train.json
~~~

8. Prepare dataset class, copy the custom_dataset.py to [dataset].py and modify the file according to the comment (Search for "Custom Dataset")

~~~
cp lib/datasets/dataset/custom_dataset.py lib/datasets/dataset/[dataset].py
~~~

9. Train

~~~
python main.py ctdet --exp_id [experiment_name] --arch resdcn_101 --dataset [dataset] --batch_size 8 --master_batch 3 --lr 1.25e-4  --gpus 0
~~~

If need resume from experiment, add `--resume`

~~~
python main.py ctdet --exp_id [experiment_name] --arch resdcn_101 --dataset [dataset] --batch_size 8 --master_batch 3 --lr 1.25e-4  --gpus 0 --resume
~~~

10. Test

~~~
python test.py ctdet --exp_id [experiment_name] --arch resdcn_101 --dataset [dataset]  --load_model ../exp/ctdet/[experiment_name]/model_best.pth --trainval
~~~

11. Try Inferencing

~~~
python demo.py ctdet --dataset [dataset] --demo [image or video] --load_model ../exp/ctdet/[experiment_name]/model_best.pth --arch resdcn_101 --debug 2
~~~


## 2. New task

You will need to add files to `src/lib/datasets/sample/`, `src/lib/datasets/trains/`, and `src/lib/datasets/detectors/`, which specify the data generation during training, the training targets, and the testing, respectively.

## 3. New architecture

- Add your model file to `src/lib/models/networks/`. The model should accept a dict `heads` of `{name: channels}`, which specify the name of each network output and its number of channels. Make sure your model returns a list (for multiple stages. Single stage model should return a list containing a single element.). The element of the list is a dict contraining the same keys with `heads`.
- Add your model in `model_factory` of `src/lib/models/model.py`.
