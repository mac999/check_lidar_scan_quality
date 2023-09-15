# check_scan_quality
3D scan data quality (version 0.1) checker

# functions
1. deviation between 3D scan data (point cloud data) and the grid of autocad drawing.
At each intersection of the grid drawn in AutoCAD, the height value of the input point group is checked, and the deviation is calculated, analyzed, and visualized.

<p align="center">
<img height="200" src="https://github.com/mac999/check_scan_quality/blob/main/doc/input1.PNG"/>
<img height="200" src="https://github.com/mac999/check_scan_quality/blob/main/doc/input2.PNG"/>
<img height="200" src="https://github.com/mac999/check_scan_quality/blob/main/doc/input3.PNG"/></br>
<img height="300" src="https://github.com/mac999/check_scan_quality/blob/main/doc/output4.PNG"/>
<img height="300" src="https://github.com/mac999/check_scan_quality/blob/main/doc/output9.PNG"/>
</p>

# usage
git clone https://github.com/mac999/check_scan_quality</br>
run autocad</br>
load grid.dwg in autocad</br>
python diff_cad_pcd.py</br>
</br>
option</br>
--input: example=sample_floor.pcd. help=input scan data file (pcd).</br>
--output: example=output. help=output excel and report(pdf) file.</br>
--config: example=config.json. help=input config.json file.</br>
--title: example=Scan Data Quality Control Report. help=title of report.</br>
--date: example=2023-09-01. help=date of report.</br>
--author: example=building points. help=maker of report</br>

# test results
<img height="200" src="https://github.com/mac999/check_scan_quality/blob/main/doc/output1.PNG"/><img height="200" src="https://github.com/mac999/check_scan_quality/blob/main/doc/output2.PNG"/><img height="200" src="https://github.com/mac999/check_scan_quality/blob/main/doc/output3.PNG"/>
<img height="200" src="https://github.com/mac999/check_scan_quality/blob/main/doc/output5.PNG"/><img height="200" src="https://github.com/mac999/check_scan_quality/blob/main/doc/output6.PNG"/><img height="200" src="https://github.com/mac999/check_scan_quality/blob/main/doc/output7.PNG"/>
<img height="200" src="https://github.com/mac999/check_scan_quality/blob/main/doc/output8.PNG"/>

# version history
0.1: draft version.</br>

# about
develop by taewook kang(laputa99999@gmail.com).
MIT license.
