# Face Mask Detection using Nvidia Deepstream

Offshoot of [project fox](https://github.com/firekind/project-fox) that detects face masks using YoloV3. Done for 8th semester project.

## Development

Use docker to setup the development environment. Take a look at `8th-sem-project.ipynb` on how to train.

## Deepstream

first, build the custom library,

```
$ cd deepstream/nvdsinfer_custom_impl_Yolo
$ CUDA_VER=10.2 make
```

then run the deepstream application using

```
$ cd .. # assuming you were in the nvdsinfer_custom_impl_Yolo folder
$ deepstream-app -c deepstream_app_config_yoloV3.txt
```
