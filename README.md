# check_scan_quality
3D scan data quality (version 0.1) checker

# functions
1. deviation between 3D scan data (point cloud data) and the grid of autocad drawing.
At each intersection of the grid drawn in AutoCAD, the height value of the input point group is checked, and the deviation is calculated, analyzed, and visualized.

<p align="center">
<img height="200" src="https://github.com/mac999/check_scan_quality/doc/input1.PNG"/>
<img height="200" src="https://github.com/mac999/check_scan_quality/doc/input2.PNG"/>
<img height="200" src="https://github.com/mac999/check_scan_quality/doc/input3.PNG"/></br>
<img height="200" src="https://github.com/mac999/check_scan_quality/doc/output1.PNG"/>
<img height="200" src="https://github.com/mac999/check_scan_quality/doc/output2.PNG"/>
<img height="200" src="https://github.com/mac999/check_scan_quality/doc/output3.PNG"/>
<img height="200" src="https://github.com/mac999/check_scan_quality/doc/output4.PNG"/>
<img height="200" src="https://github.com/mac999/check_scan_quality/doc/output5.PNG"/>
<img height="200" src="https://github.com/mac999/check_scan_quality/doc/output6.PNG"/>
<img height="200" src="https://github.com/mac999/check_scan_quality/doc/output7.PNG"/>
<img height="200" src="https://github.com/mac999/check_scan_quality/doc/output8.PNG"/>
<img height="200" src="https://github.com/mac999/check_scan_quality/doc/output9.PNG"/>
</p>

# usage
git clone https://github.com/mac999/check_scan_quality
python diff_cad_pcd.py

# version history
0.1: draft version.</br>

# about
develop by taewook kang(laputa99999@gmail.com).
MIT license.
