# title: diff cad pcd
# author: Taewook Kang
# date: 2023.8.1
# description: difference between cad and pcd (scan data)
# license: MIT
# 
import os, math, argparse, json, traceback, numpy as np, pandas as pd, trimesh, laspy
import pyautocad, open3d as o3d, seaborn as sns, win32com.client, pythoncom
from jakteristics import FEATURE_NAMES, extension, las_utils, utils
import matplotlib.pyplot as plt
from scipy.spatial import distance
from tqdm import trange, tqdm
from math import pi
from fpdf import FPDF

_config = {}
_option = ""	# 'planarity | verticality | features'

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

def get_ref_points_from_acad(point_layer_name):
	# Connect to AutoCAD
	acad = pyautocad.Autocad(create_if_not_exists=True)

	# Get all existing line entities from the current drawing
	points = [entity for entity in acad.iter_objects() if entity.EntityName == 'AcDbPoint' and entity.Layer == point_layer_name]

	if not points:
		print("No point entities found in the drawing.")
		return

	# Calculate intersection points
	ints = []
	for point in points:
		x = point.InsertionPoint[0]
		y = point.InsertionPoint[1]
		z = point.InsertionPoint[2]
		ints.append((x, y, z))

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

def output_acad_diff_model(diffs):
	global _config, _option

	if len(diffs) == 0:
		return

	add_acad_layer("scan_diff")

	try:		
		acad = win32com.client.Dispatch("AutoCAD.Application")
		acad_model = acad.ActiveDocument.ModelSpace

		color_schema = _config['color_schema.height']
		hatchs = add_acad_hatch(acad_model, color_schema)

		max = get_max_diff(diffs)
		for index, diff in enumerate(diffs):
			ID = int(diff[0])
			pt = diff[1:4]
			p1 = APoint(pt[0], pt[1], pt[2])

			value = math.fabs(diff[4]) / max
			diff_index = get_color_schema_index(color_schema, value)	# https://gohtx.com/acadcolors.php

			out_loop = []
			ent = acad_model.AddPoint(p1)

	except:
		traceback.print_exc()
		pass

def output_acad_diff(diffs):
	global _config, _option

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
		offset_x = radius * 1.5			# TBD. option
		offset_y = radius
		for index, diff in tqdm(enumerate(diffs), desc='output cad'):
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

def convert_las_to_pcd(infile, outfile):
	pipe_data = {
		"pipeline": [
			infile, 
			{
			"type":"filters.transformation",
			"matrix":"1  0  0  0  0  1  0  0  0  0  1  0  0  0  0  1"
			}, 
			{
			"type":"writers.pcd",
			"filename":outfile
			} 
		]
	}

	pdal_pipeline = str(Path(__file__).parent) + "/las_to_pcd_pipeline.json"
	with open(pdal_pipeline, 'w') as f:
		json.dump(pipe_data, f)

	cmd = ["pdal", "pipeline", pdal_pipeline]

	ret = subprocess.call(cmd) 
	print(ret)

def load_point_cloud(file_path):
	pcd = None

	fname, ext = os.path.splitext(file_path)
	if ext == '.las':
		xyz = las_utils.read_las_xyz(file_path)
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(xyz)
	else:
		pcd = o3d.io.read_point_cloud(file_path + ".pcd")
	return pcd

def compute_signed_distance_and_closest_goemetry(target_mesh, query_points): # http://www.open3d.org/docs/latest/tutorial/geometry/distance_queries.html
	''' closest_points = scene.compute_closest_points(query_points)
	distance = np.linalg.norm(query_points - closest_points['points'].numpy(),
							  axis=-1)
	rays = np.concatenate([query_points, np.ones_like(query_points)], axis=-1)
	intersection_counts = scene.count_intersections(rays).numpy()
	is_inside = intersection_counts % 2 == 1
	distance[is_inside] *= -1
	return distance, closest_points['geometry_ids'].numpy() '''

	dataset = trimesh.proximity.closest_point(target_mesh, query_points) # https://github.com/mikedh/trimesh/issues/1116
	return dataset

def compare_pcd_model_diff(target_mesh, pcd):
	global _config

	diffs = []
	try:
		for index, vertex in tqdm(enumerate(pcd.points), desc='calculate model difference'):
			closest_points, diff_distances, indics = compute_signed_distance_and_closest_goemetry(target_mesh, [vertex])
			if len(closest_points) == 0:
				continue
			
			min_index = np.argmin(diff_distances)
			diff_distance = diff_distances[min_index]
			min_point = closest_points[min_index]
			if diff_distance > _config['check_model']['min_distance']:
				continue

			diffs.append((min_point[0], min_point[1], min_point[2], diff_distance))

		diffs = np.array(diffs)
	except:
		traceback.print_exc()
		pass

	return diffs

def compare_pcd_diff(target_mesh, pcd, ints):
	global _config

	try:
		z_differences = []

		pcd_tree = o3d.geometry.KDTreeFlann(pcd)

		for index, intersection_point in tqdm(enumerate(ints), desc='calculate difference'):
			if intersection_point == None:
				continue
			_, idx, _ = pcd_tree.search_knn_vector_3d(intersection_point, 3)
			if idx == None or len(idx) == 0:
				continue

			# if math.fabs(intersection_point[0] - 2.5) < 0.001 and math.fabs(intersection_point[1] - 1.25) < 0.001:
			# 	index = index

			find_pt_pcd = pcd.points[idx[0]]
			dist = get_2d_distance(find_pt_pcd, intersection_point)
			if dist > _config['check_height']['distance_tolerance']:
				continue

			closest_point_model = None
			closest_points = compute_signed_distance_and_closest_goemetry(target_mesh, intersection_point)
			for cp in closest_points[0]:
				closest_point_model = tuple(cp)
				diff_distance = distance.euclidean(intersection_point, closest_point_model)

			z_nearest_pcd = find_pt_pcd[2]

			z_intersection_model = closest_point_model[2]
			z_intersection_model += _config['check_height']['base_height']

			z_difference = z_nearest_pcd - z_intersection_model
			z_differences.append((index + 1, intersection_point[0], intersection_point[1], intersection_point[2], z_difference))

		z_differences = np.array(z_differences)
	except:
		traceback.print_exc()
		pass

	return z_differences    

def get_plane_mesh(z = 0.0):
	plane = trimesh.creation.box(extents=[100000, 100000, 1.0])
	plane = plane.apply_translation([0, 0, -0.5])
	return plane

def get_check_points_from_acad(config):
	ints = get_intersects_from_acad(config['check_height']['grid_layer'])
	if 'point_layer' in config['check_height']:
		ints2 = get_ref_points_from_acad(config['check_height']['point_layer'])
		if ints2 != None and len(ints2):
			ints.extend(ints2)
	return ints

def check_flatness(input_fname):
	global _config
	ints = get_check_points_from_acad(_config)

	mesh = get_plane_mesh(z = 0.0)
	pcd = load_point_cloud(input_fname)
	diffs = compare_pcd_diff(mesh, pcd, ints)

	return pcd, ints, diffs

def check_model(input_fname, model_fname):
	mesh = trimesh.load_mesh(model_fname)	# issue: rotated by 90 degree (x-axis)
	angle = math.pi / 2
	direction = [1, 0, 0]
	center = [0, 0, 0]
	rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)
	mesh.apply_transform(rot_matrix)

	pcd = load_point_cloud(input_fname)
	diffs = compare_pcd_model_diff(mesh, pcd)

	return pcd, diffs

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

def output_pcd(input_path, output_path, diffs):
	try:
		'''
		pcd = o3d.geometry.PointCloud()
		# xyz = np.concatenate((diffs[:, 1:3], diffs[:, 4:5]), axis=1)
		pcd.points = o3d.utility.Vector3dVector(diffs[:, 0:3])
		o3d.io.write_point_cloud(output_fname, pcd)
		'''
		FEATURE_NAMES = [
			"tan_x",
			"tan_y",
			"tan_z",		
			"diff"
		]
		# features = np.concatenate((diffs[:, 0:3], diffs[:, 3:4]), axis=1)
		
		las_utils.write_with_extra_dims(input_path, output_path, diffs, FEATURE_NAMES)		
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
	global _config, _option

	parser = argparse.ArgumentParser()
	# parser.add_argument('--input', default='.\\sample_floor.pcd', help='input scan data file (pcd).')
	# parser.add_argument('--model', default='', help='input model file (obj, stl, ply, off).')
	parser.add_argument('--input', default='.\\sample_floor.las', help='input scan data file (pcd).')
	parser.add_argument('--model', default='simple_mesh.obj', help='input model file (obj, stl, ply, off).')
	parser.add_argument('--output', default='.\\output', help='output excel and report(pdf) file.')
	# parser.add_argument('--option', default='planarity', help='planarity | verticality | features | model')
	parser.add_argument('--option', default='model', help='planarity | verticality | features | model')
	parser.add_argument('--config', default='.\\config.json', help='input config.json file.')
	parser.add_argument('--title', default='Scan Data Quality Control Report', help='title of report.')
	parser.add_argument('--date', default='2023-09-0+1', help='date of report.')
	parser.add_argument('--author', default='building points', help='maker of report')

	args = parser.parse_args()
	print('args: ', args)
	try:
		load_config(args.config)

		_option = args.option

		if args.option == 'planarity':
			pcd, ints, diffs = check_flatness(args.input)
			output_acad_diff(diffs)
			output_excel(args.output + ".xlsx", pcd, diffs)
			output_report(args.output + ".pdf", args.title, args.author, args.date, pcd, diffs)
		elif args.option == 'model':
			pcd, diffs = check_model(args.input, args.model)
			output_pcd(args.input, args.output + ".las", diffs)
			# output_acad_diff_model(diffs)
			# output_excel(args.output + ".xlsx", pcd, diffs)
			# output_report(args.output + ".pdf", args.title, args.author, args.date, pcd, diffs)
		else:
			pass

	except:
		traceback.print_exc()

if __name__ == "__main__":
	main()
