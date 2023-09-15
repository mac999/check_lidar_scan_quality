# title: diff cad pcd
# author: Taewook Kang
# date: 2023.8.1
# description: difference between cad and pcd (scan data)
# license: MIT
# 
import os, math, argparse, json, traceback, numpy as np, pandas as pd
import pyautocad, lorem, open3d as o3d, seaborn as sns, win32com.client, pythoncom
from math import pi
import matplotlib.pyplot as plt
from fpdf import FPDF

_config = {}

def load_config(config_fname):
	global _config
	with open(config_fname) as json_file:
		_config = json.load(json_file)

def find_intersection_points(lines):
	intersection_points = []
	for i in range(len(lines)):
		for j in range(i + 1, len(lines)):
			line1 = lines[i]
			line2 = lines[j]
			
			intersections = line1.IntersectWith(line2, 0) # , pyautocad.constants.acExtendThisEntity)
			
			intersection_points.extend(intersections)
	return intersection_points

def get_2d_distance(pt1, pt2):
	x1 = pt1[0]
	y1 = pt1[1]
	x2 = pt2[0]
	y2 = pt2[1]
	return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def get_intersects_from_acad(layer_name):
	# Connect to AutoCAD
	acad = pyautocad.Autocad(create_if_not_exists=True)

	# Get all existing line entities from the current drawing
	lines = [entity for entity in acad.iter_objects() if entity.EntityName == 'AcDbLine' and entity.Layer == layer_name]

	if not lines:
		print("No line entities found in the drawing.")
		return

	# Calculate intersection points
	intersection_points = find_intersection_points(lines)

	ints = []
	if intersection_points:
		index = 0
		while index < len(intersection_points):
			x = intersection_points[index]
			y = intersection_points[index + 1]
			z = intersection_points[index + 2]

			ints.append((x, y, z))
			print(f"Intersection Point {index / 3}: X={x}, Y={y}, Z={z}")
			index += 3
	else:
		print("No intersection points found.")

	return ints

def add_acad_layer(layer_name):
	acad = pyautocad.Autocad(create_if_not_exists=True)	

	try:
		layer = acad.doc.Layers.Add(layer_name)
		layer.color = 40
		layer.linetype = "CONTINUOUS"
		acad.doc.ActiveLayer = layer
	except:
		pass

def add_acad_hatch(acad_model, color_schema):
	global _config

	try:
		hatchs = []
		for c in color_schema:
			index = c['index']
			start = c['start']
			end = c['end']

			hatch = acad_model.AddHatch(0, "SOLID", True) # "ANSI37", True,)	# https://www.cadforum.cz/forum/uploads/81/AC69hatch.pdf
			hatch.PatternScale = 0.1
			hatch.color = index
			hatchs.append(hatch)
		return hatchs
	except:
		traceback.print_exc()
		pass
	return None

def get_color_schema_index(color_schema, value):
	for index, c in enumerate(color_schema):
		start = c['start']
		end = c['end']
		if value >= start and value < end:
			return index
	return -1

def APoint(x, y, z = 0):
	return win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (x, y, z))

def ADouble(xyz):
	return win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (xyz))

def variants(object):
	return win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_DISPATCH, (object))

def get_max_diff(diffs):
	min = math.fabs(diffs[:, 4].min())
	max = math.fabs(diffs[:, 4].max())
	if min > max:
		max = min
	return max

def output_acad_diff(pcd, ints, diffs):
	global _config

	if len(diffs) == 0:
		return

	add_acad_layer("scan_diff")

	try:		
		acad = win32com.client.Dispatch("AutoCAD.Application")
		acad_model = acad.ActiveDocument.ModelSpace

		color_schema = _config['color_schema.height']
		hatchs = add_acad_hatch(acad_model, color_schema)

		max = get_max_diff(diffs)
		
		text_height = _config['check_height']['text_height']
		radius = _config['check_height']['circle_radius']
		offset_x = radius * 1.5
		offset_y = radius
		for index, diff in enumerate(diffs):
			ID = int(diff[0])
			pt = diff[1:4]
			p1 = APoint(pt[0], pt[1])
			p2 = APoint(pt[0] + offset_x, pt[1] + offset_y)
	
			output = f'ID:{ID}'
			if text_height > 0.0:
				acad_model.AddLine(p1, p2)
				acad_model.AddText(output, p2, text_height)
			p2 = APoint(pt[0] + offset_x, pt[1] - offset_y)

			output = ''
			if _config['check_height']['show_coord']:
				output += f'({pt[0]:.3f},{pt[1]:.3f})'
			if _config['check_height']['show_height']:
				output += f'[{diff[4]:.3f}]'
			if len(output):
				acad_model.AddText(output, p2, text_height / 2.0)

			value = math.fabs(diff[4]) / max
			diff_index = get_color_schema_index(color_schema, value)	# https://gohtx.com/acadcolors.php

			out_loop = []
			circle = acad_model.AddCircle(p1, radius)
			out_loop.append(circle)
			outer = variants(out_loop)
			hatchs[diff_index].AppendOuterLoop(outer)

	except:
		traceback.print_exc()
		pass

def load_point_cloud(file_path):
	pcd = o3d.io.read_point_cloud(file_path)
	return pcd

def compare_pcd_diff(pcd, ints):
	global _config

	z_differences = []

	pcd_tree = o3d.geometry.KDTreeFlann(pcd)

	for index, intersection_point in enumerate(ints):
		_, idx, _ = pcd_tree.search_knn_vector_3d(intersection_point, 3)
		if idx == None or len(idx) == 0:
			continue

		# if math.fabs(intersection_point[0] - 2.5) < 0.001 and math.fabs(intersection_point[1] - 1.25) < 0.001:
		# 	index = index

		find_pt = pcd.points[idx[0]]
		dist = get_2d_distance(find_pt, intersection_point)
		if dist > _config['check_height']['distance_tolerance']:
			continue

		z_intersection = intersection_point[2]
		z_intersection += _config['check_height']['base_height']
		z_nearest = find_pt[2]

		z_difference = z_nearest - z_intersection
		z_differences.append((index + 1, intersection_point[0], intersection_point[1], intersection_point[2], z_difference))

	z_differences = np.array(z_differences)
	return z_differences    

def check_flatness(input_fname):
	global _config
	try:
		ints = get_intersects_from_acad(_config['check_height']['layer'])

		pcd = load_point_cloud(input_fname)
		diffs = compare_pcd_diff(pcd, ints)

		return pcd, ints, diffs
	except:
		traceback.print_exc()
		pass
	return None, None, None

def output_excel(output_fname, pcd, diffs):
	try:
		df = pd.DataFrame(columns=['ID', 'X', 'Y', 'Z', 'Diff'])
		for index, diff in enumerate(diffs):
			ID = diff[0]
			x = diff[1]
			y = diff[2]
			z = diff[3]
			d = diff[4]
			print(f'point {ID} = ({x:.2f}, {y:.2f}, {z:.2f}), height diff = {d:.3f}')

			df = pd.concat([df, pd.DataFrame([[ID, x, y, z, d]], columns=['ID', 'X', 'Y', 'Z', 'Diff'])])

		df.to_excel(output_fname, index=False, sheet_name='Difference')
	except:
		traceback.print_exc()
		pass

def get_count_in_diffs(diffs, start, end):
	max = get_max_diff(diffs)

	count = 0
	for index, diff in enumerate(diffs):
		d = math.fabs(diff[4]) / max
		if d >= start and d < end:
			count += 1
	return count

def output_report(output_fname, title, author, date, pcd, diffs):
	global _config

	try:
		# PDF class
		ch = 8
		class PDF(FPDF):
			def __init__(self):
				super().__init__()
			def header(self):
				self.set_font('Arial', '', 12)
				self.cell(0, 8, '3D Scan Data Quality Control Report', 0, 1, 'C')
			def footer(self):
				self.set_y(-15)
				self.set_font('Arial', '', 12)
				self.cell(0, 8, f'Page {self.page_no()}', 0, 0, 'C')

		# page
		pdf = PDF()
		pdf.add_page()
		pdf.set_font('Arial', 'B', 16)
		pdf.cell(w=0, h=20, txt=title, ln=1)
		pdf.set_font('Arial', '', 12)
		pdf.cell(w=30, h=ch, txt="Date: ", ln=0)
		pdf.cell(w=30, h=ch, txt=date, ln=1)
		pdf.cell(w=30, h=ch, txt="Author: ", ln=0)
		pdf.cell(w=30, h=ch, txt=author, ln=1)
		pdf.ln(ch)

		# Statistics
		col = ['No', 'Start', 'End', 'Color', 'Count', 'Ratio']
		total = len(diffs)
		df = pd.DataFrame(columns=col)
		for index, c in enumerate(_config['color_schema.height']):
			color = c['name']
			start = c['start']
			end = c['end']

			count = get_count_in_diffs(diffs, start, end)
			ratio = count / total * 100.0
			df = pd.concat([df, pd.DataFrame([[index + 1, start, end, color, count, ratio]], columns=col)])

		pdf.cell(w=100, h=ch, txt="Table. Height Difference Statistics between CAD and Scan data", ln=1)
		pdf.set_font('Arial', 'B', 12)
		pdf.cell(w=15, h=ch, txt='No', border=1, ln=0, align='C')
		pdf.cell(w=30, h=ch, txt='Start', border=1, ln=0, align='C')
		pdf.cell(w=30, h=ch, txt='End', border=1, ln=0, align='C')
		pdf.cell(w=30, h=ch, txt='Color', border=1, ln=0, align='C')
		pdf.cell(w=30, h=ch, txt='Count', border=1, ln=0, align='C')
		pdf.cell(w=30, h=ch, txt='Ratio(%)', border=1, ln=1, align='C')
		pdf.set_font('Arial', '', 10)
		for i in range(len(df)):
			pdf.cell(w=15, h=ch, txt=str(df['No'].iloc[i]), border=1, ln=0, align='C')
			start = "{:.5f}".format(df['Start'].iloc[i])
			pdf.cell(w=30, h=ch, txt=start, border=1, ln=0, align='R')
			end = "{:.5f}".format(df['End'].iloc[i])
			pdf.cell(w=30, h=ch, txt=end, border=1, ln=0, align='R')
			color = df['Color'].iloc[i]
			pdf.cell(w=30, h=ch, txt=color, border=1, ln=0, align='R')
			count = "{:d}".format(df['Count'].iloc[i])
			pdf.cell(w=30, h=ch, txt=count, border=1, ln=0, align='R')
			ratio = "{:.2f}".format(df['Ratio'].iloc[i])
			pdf.cell(w=30, h=ch, txt=ratio, border=1, ln=1, align='R')

		pdf.ln(ch)

		fig, ax = plt.subplots(1,1, figsize = (6, 4))
		sns.barplot(data =  df, x = 'No', y = 'Count', color='orange', ax = ax)
		pdf.set_font('Arial', 'B', 12)
		plt.title("Figure. Histogram of Height Difference")
		plt.savefig('./diff1.png', 
				transparent=False,  
				facecolor='white', 
				bbox_inches="tight")
		
		# pdf.multi_cell(w=0, h=5, txt=lorem.paragraph())
		pdf.image('./diff1.png', x = 10, y = None, w = 100, h = 0, type = 'PNG', link = '')
		# pdf.ln(ch)
		# pdf.multi_cell(w=0, h=5, txt=lorem.paragraph())
		pdf.ln(ch)

		# Table contents
		df = pd.DataFrame(columns=['ID', 'X', 'Y', 'Z', 'Diff'])
		for index, diff in enumerate(diffs):
			ID = diff[0]
			x = diff[1]
			y = diff[2]
			z = diff[3]
			d = diff[4]

			df = pd.concat([df, pd.DataFrame([[ID, x, y, z, d]], columns=['ID', 'X', 'Y', 'Z', 'Diff'])])

		pdf.set_font('Arial', 'B', 12)
		pdf.cell(w=100, h=ch, txt="Table. Height Difference", ln=1)
		pdf.cell(w=20, h=ch, txt='ID', border=1, ln=0, align='C')
		pdf.cell(w=30, h=ch, txt='Diff', border=1, ln=1, align='C')
		pdf.set_font('Arial', '', 10)
		for i in range(len(df)):
			ID = "{:.0f}".format(df['ID'].iloc[i])
			pdf.cell(w=20, h=ch, txt=ID, border=1, ln=0, align='C')
			diff = "{:.5f}".format(df['Diff'].iloc[i])
			pdf.cell(w=30, h=ch, txt=diff, border=1, ln=1, align='R')

		pdf.output(output_fname)
	except:
		traceback.print_exc()
		pass

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', default='.\\sample_floor.pcd', help='input scan data file (pcd).')
	parser.add_argument('--output', default='.\\output', help='output excel and report(pdf) file.')
	parser.add_argument('--config', default='.\\config.json', help='input config.json file.')
	parser.add_argument('--title', default='Scan Data Quality Control Report', help='title of report.')
	parser.add_argument('--date', default='2023-09-01', help='date of report.')
	parser.add_argument('--author', default='building points', help='maker of report')

	args = parser.parse_args()
	print('args: ', args)
	try:
		load_config(args.config)

		pcd, ints, diffs = check_flatness(args.input)
		output_acad_diff(pcd, ints, diffs)
		output_excel(args.output + ".xlsx", pcd, diffs)
		output_report(args.output + ".pdf", args.title, args.author, args.date, pcd, diffs)

	except:
		traceback.print_exc()

if __name__ == "__main__":
	main()