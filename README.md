# SNU Computer Vision Project

## Data
input images are in data/dataset.

output images for each python file are in results/{vanishing_point, environment_map, sun_position}.

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

### Running sun position estimation
```
python sun_position_estimation.py
```
