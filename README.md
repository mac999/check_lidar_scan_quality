# check scan quality
3D scan data quality (version 0.1) checker

<p align="center">
<img height="300" src="https://github.com/mac999/check_scan_quality/blob/main/doc/Animation.gif"/>
</p>

# functions
Introducing an open source that checks the quality of PCD (Point Cloud Data), 3D scan data from LiDAR, Drone-based photogrammetry. It is linked to AutoCAD, and if there is a grid in the drawing, the intersection point is automatically calculated, compared to the given point cloud height at that point, and analyzed. Errors are output to drawings, Excel, and reports according to a predefined color scheme. Options can be modified with config.json. Currently, I only developed the planarity check, and plan to update it periodically in the future. It provides various options so you can modify it and use it.</br></br>

1. deviation between 3D scan data (point cloud data) and the grid of autocad drawing.
At each intersection of the grid drawn in AutoCAD, the height value of the input point group is checked, and the deviation is calculated, analyzed, and visualized.
<p align="center">
<img height="200" src="https://github.com/mac999/check_scan_quality/blob/main/doc/input1.PNG"/>
<img height="200" src="https://github.com/mac999/check_scan_quality/blob/main/doc/input2.PNG"/>
<img height="200" src="https://github.com/mac999/check_scan_quality/blob/main/doc/input3.PNG"/></br>
<img height="300" src="https://github.com/mac999/check_scan_quality/blob/main/doc/output4.PNG"/>
<img height="300" src="https://github.com/mac999/check_scan_quality/blob/main/doc/output9.PNG"/>
</p>

2. comparison between Model (mesh) and 3D scan data (point cloud data).
<p align="center">
<img height="300" src="https://github.com/mac999/check_scan_quality/blob/main/doc/model1.JPG"/>
<img height="300" src="https://github.com/mac999/check_scan_quality/blob/main/doc/model2.JPG"/>
</p>

# usage
download executable program in <a href="https://drive.google.com/drive/folders/1r_WErxpp5JMlvKBU-88S5SogaCY37fY5?usp=drive_link">here</a>.</br>
or</br>
run terminal(cmd)</br>
git clone https://github.com/mac999/check_scan_quality</br>
conda install --file conda_requirements.txt</br>
run autocad</br>
load grid.dwg in autocad</br>
python diff_cad_pcd.py</br>
</br>
option</br>
--input: example=sample_floor.pcd. help=input scan data file (pcd).</br>
--model: example=simple_mesh.obj. help=input model file (obj, stl, ply, off).</br>
--output: example=output. help=output excel and report(pdf) file.</br>
--config: example=config.json. help=input config.json file.</br>
--title: example=Scan Data Quality Control Report. help=title of report.</br>
--date: example=2023-09-01. help=date of report.</br>
--author: example=building points. help=maker of report</br>
</br>
ex. python diff_cad_pcd.py --input sample_floor.pcd</br>
<img height="150" src="https://github.com/mac999/check_scan_quality/blob/main/doc/run.PNG"/>

# config
this program supports variable option to visualize the results in cad, report.</br>
```json
{
    "name": "check scan quality config",
    "version": "1.0.0",
    "description": "check scan quality config",
    "author": "mac999",
    "email": "",
    "check_height": {
        "layer": "0",
        "base_height": 0.0,
        "distance_tolerance": 0.02,
        "circle_radius": 0.1,
        "text_height": 0.05,
        "show_coord": true,
        "show_height": true
    },
    "check_model": {
        "min_distance": 10.0
    },
    "color_schema.height": [
        {
            "name": "green",
            "index": 3,
            "start": 0.0,
            "end": 0.2
        }, 
        {
            "name": "cyan",
            "index": 4,
            "start": 0.2,
            "end": 0.4
        }, 
        {
            "name": "yellow",
            "index": 2,
            "start": 0.4,
            "end": 0.7
        }, 
        {
            "name": "red",
            "index": 1,
            "start": 0.7,
            "end": 1.01
        }
    ]
}
```

# test results
<img height="200" src="https://github.com/mac999/check_scan_quality/blob/main/doc/output1.PNG"/><img height="200" src="https://github.com/mac999/check_scan_quality/blob/main/doc/output2.PNG"/><img height="200" src="https://github.com/mac999/check_scan_quality/blob/main/doc/output3.PNG"/>
<img height="200" src="https://github.com/mac999/check_scan_quality/blob/main/doc/output5.PNG"/><img height="200" src="https://github.com/mac999/check_scan_quality/blob/main/doc/output6.PNG"/><img height="200" src="https://github.com/mac999/check_scan_quality/blob/main/doc/output7.PNG"/>
<img height="200" src="https://github.com/mac999/check_scan_quality/blob/main/doc/output8.PNG"/>

# version history
0.1: draft version.</br>
0.2: under developing, updating difference between PCD and mesh object.</br>

# future plan
will support</br>
Features(curvature ...) generation from PCD</br>
LiDAR file format support such as E57</br>
BIM comparison and analysis</br>

# license
develop by taewook kang(laputa99999@gmail.com).
MIT license.
