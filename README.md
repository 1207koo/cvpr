# SNU Computer Vision Project

## Data
input images are in `data/dataset`.

input images should be Manhattan world images with the sky contained.

output images for each python file are in `results/{vanishing_point, environment_map, sun_position}`.

## Run
### Run calculatng camera parameters
```
python camera_parameter.py
```
This will run vanishing point detection and calculate camera parameters. Result values(position of vanishing points and camera parameter) will be printed and result image(line detection and vanishing point detection) will be saved as file.


You can change directory or dataset by changing `datadir, resultdir, imagename` in the code.

The result image will be saved in `resultdir+'/vanishing_point/'+imagename`

`imagename` also can be regex string, such as `"/*.jpg"`


### Running environment mapping
```
python environment_mapping.py
```
This will run enrionment mapping. Output file is a single image, which is an environment map of input image.

Regarding the input file directory is same as above.

The result image will be saved in `resultdir+'/environment_map/'+imagename`

### Running sun position estimation
```
python sun_position_estimation.py
```
This will run sun position estimation. Output files are two images, which are sky segmentation result and sun position likelihood.

Regarding the input file directory is same as above.

The result images will be saved in `resultdir+'/sun_position/'` directory, and file names are `imagename+'_sky', imagename+'_sun'`, respectively.
