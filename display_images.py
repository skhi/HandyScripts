from __future__ import division

import numpy as np
from PIL import Image
from bokeh.plotting import figure, show, output_file
from bokeh.document import Document
from bokeh.embed import file_html
from bokeh.layouts import gridplot
from bokeh.models.glyphs import Circle, Line
from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, Range1d
from bokeh.resources import INLINE
from datetime import date
from random import randint
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.models.widgets.tables import StringFormatter
from bokeh.layouts import widgetbox
from bokeh.layouts import row
from bokeh.models import Label
from bokeh.io import show, save
from bokeh.palettes import Viridis3
import matplotlib.pyplot as plt
import itertools
import optparse
import glob
import os
import pandas_ml as pdml
import cStringIO
import sys



parser = optparse.OptionParser()
parser.add_option('-f', '--file_name',
                  dest = "file_name",
                  type = "string",
                  help = "Indicate a file with a list of images in it" )

parser.add_option('-p', '--path',
                  dest = "path",
                  type = "string",
                  help = "Indicate a path to the remote image storage" )

parser.add_option('-t', '--type',
                  dest      = "type",
                  type      = "string",
                  default   = "scale_width",
                  help      = "Indicate a plotting type: fixed, stretch_both, scale_width, scale_height, scale_both. Default is 'scale_width'" )

parser.add_option('-n', '--name',
                  dest = "name",
                  type = "string",
                  help = "Indicate a name for the html file" )

parser.add_option('-e', '--ext',
                  dest      = "ext",
                  type      = "string",
                  default   = ".tif",
                  help      = "Indicate an extention of image files. Default is .tif. For all three image collections (raw, mask, prediction), it should be uniform" )

parser.add_option('-s', '--summary',
                  dest      = "summary",
                  type      = "string",
                  default   = "with",
                  help      = "Indicate if you want the output to be only suumary ('only' vs 'with'): by default the output comes with summary" )

(args, remaining_args) = parser.parse_args()



def get_image_list_from_file(name):
    image_list = []
    name_list = []
    with open(name, 'r') as f:
        for line in f:
            image_list.append(line.rstrip('\n').split(" ")[0])
            name_list.append(line.rstrip('\n').split(" ")[1])
    return (image_list,name_list)

def get_image_list_from_dir(dir, ext):
    print dir+"*"+ext
    image_list = glob.glob(dir+os.sep+"*"+ext)
    return image_list


def read_image(input, img_name):
    raw_img     = Image.open(input).convert('RGBA')
    xdim, ydim  = raw_img.size
    mod_img     = np.empty((ydim, xdim), dtype=np.uint32)
    view        = mod_img.view(dtype=np.uint8).reshape((ydim, xdim, 4))
    view[:,:,:] = np.flipud(np.asarray(raw_img))
    name        = input.split('/')[-1].split('.')[0]
    name = name + "    " + img_name
    return (view[:,:,:], name, xdim, ydim)


def fig(x, y, label, name, x_axis_name, y_axis_name, x_range, y_range, size=7, color=Viridis3[0]):
    plot = figure(x_range = x_range, y_range = y_range, title=name)
    plot.circle(x, y, size=size, color=color)
    plot.xaxis.axis_label = x_axis_name
    plot.yaxis.axis_label = y_axis_name
    plot.add_layout(label)
    return plot


def cit(text, x, y,):
    citation = Label(x=x,
                     y=y,
                     x_units='screen',
                     y_units='screen',
                     text=str("average: ")+text,
                     render_mode='css',
                     border_line_color='white',
                     border_line_alpha=1.0,
                     background_fill_color='white',
                     background_fill_alpha=1.0)
    return citation

def prepare_image(input):

    # Display the 32-bit RGBA image
    fig = figure(title=input[1],
                 x_range=(0,input[2]),
                 y_range=(0,input[3]))

    fig.image_rgba( image=[input[0]],
               x=0,
               y=0,
               dw=input[2],
               dh=input[3])

    return fig

def match_images(my_image_list, all_images, type):
    matched_images = []
    for i in all_images:
        matched = False
        if type in i:
            for j in my_image_list:
                if str(j) == i.split('/')[-1].split('_')[1].split('.')[0]:
                    matched = True
                    break
            if matched is True:
                matched_images.append(i)
    return matched_images




my_image_list, my_image_name  = get_image_list_from_file(args.file_name)
all_image_list = get_image_list_from_dir(args.path, args.ext)

mask_list = match_images(my_image_list,all_image_list,'mask' )
raw_list  = match_images(my_image_list,all_image_list,'raw' )
pred_list = match_images(my_image_list,all_image_list,'prediction' )


if len(mask_list) == len(pred_list) == len(raw_list) == len(my_image_name):
    print "raw==mask==predicted==names"
else:
    print "raw!=mask!=predicted!=names: Something is wrong"
    sys.exit(0)


output_file(args.name)

raw_img =[]
mask_img=[]
pred_img=[]

for i, j, k, n in zip(raw_list,mask_list,pred_list,my_image_name):
    raw_img.append( prepare_image(read_image(i, n ) ) )
    mask_img.append( prepare_image(read_image(j, n ) ) )
    pred_img.append( prepare_image(read_image(k, n ) ) )


result = [None]*(len(raw_img)+len(mask_img)+len(pred_img)+len(pred_img) )


dec         = 4
box         = []
fig_box     = []
f1_score    = []
sensitivity = []
specificity = []
accuracy    = []
# count a number of images in the file in order to reshape the numpy array accordingly
image_num = 0
for  m,p in zip(mask_list,pred_list):
    image_num = image_num + 1
    tmp_file = open("tmp.txt", 'w')
    m_img = np.array(Image.open(m), dtype = np.float32)
    p_img = np.array(Image.open(p), dtype = np.float32)
    pdml_cm = pdml.ConfusionMatrix( m_img.ravel(), p_img.ravel() )
    sys.stdout = tmp_file
    print  >> pdml_cm.print_stats()

    tmp_file.close()

    data = dict( name=[], data=[] )
    with open("tmp.txt", 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if "TP:" in line:
                data["name"].append("True Positive")
                data["data"].append(float(line.split(' ')[1]))
            elif "TN:" in line:
                data["name"].append("True Negative")
                data["data"].append(float(line.split(' ')[1]))
            elif "FP:" in line:
                data["name"].append("False Positive")
                data["data"].append(float(line.split(' ')[1]))
            elif "FN:" in line:
                data["name"].append("False Negative")
                data["data"].append(float(line.split(' ')[1]))
            elif "ACC:" in line:
                data["name"].append("Accuracy")
                data["data"].append(round( float(line.split(' ')[1]), dec) )
                accuracy.append(round( float(line.split(' ')[1]), dec) )
            elif "F1_score:" in line:
                data["name"].append("F1 Score")
                data["data"].append(round( float(line.split(' ')[1]), dec) )
                f1_score.append(round( float(line.split(' ')[1]), dec) )
            elif "TPR:" in line:
                data["name"].append("Sensitivity")
                data["data"].append(round( float(line.split(' ')[1]), dec) )
                sensitivity.append(round( float(line.split(' ')[1]), dec) )
            elif "TNR:" in line:
                data["name"].append("Specificity")
                data["data"].append(round( float(line.split(' ')[1]), dec) )
                specificity.append(round( float(line.split(' ')[1]), dec) )


    source = ColumnDataSource(data)

    columns = [
               TableColumn(field="name", title="measure"),
               TableColumn(field="data", title="value")
               ]
    data_table = DataTable(source=source, columns=columns, width=260, height=300)

    box.append( widgetbox(data_table))

# define plots for sens, spec, f1_score
image_list = list(range(image_num))

# calculate averages

sum_sens  = sum(sensitivity)
sum_spec  = sum(specificity)
sum_f1sc  = sum(f1_score)
sum_accu  = sum(accuracy)


# sensitivity
sens_cit  = cit(str(round(float(sum_sens/image_num), 2)), 80, 90,)
sens_plot = fig(image_list, sensitivity, sens_cit, "sensitivity", "images", "value", x_range=[-1, image_num+1], y_range=[0,1.1])

#specificity
spec_cit  = cit(str(round(float(sum_spec/image_num), 2)), 80, 90,)
spec_plot = fig(image_list, specificity, spec_cit, "specificity", "images", "value", x_range=[-1, image_num+1], y_range=[0,1.1])

#f1 score
f1sc_cit  = cit(str(round(float(sum_f1sc/image_num), 2)), 80, 90,)
f1sc_plot = fig(image_list, f1_score, f1sc_cit,  "f1 score", "images", "value", x_range=[-1, image_num+1], y_range=[0,1.1])

# accuracy
accu_cit  = cit(str(round(float(sum_accu/image_num), 2)), 80, 90,)
accu_plot = fig(image_list, accuracy, accu_cit,  "accuracy", "images", "value", x_range=[-1, image_num+1], y_range=[0,1.1])




result_np = []
if args.summary == "only":
    result_np = np.array([accu_plot,sens_plot,spec_plot,f1sc_plot]).reshape(1,4)

elif args.summary == "with":
    result[::4]  = raw_img
    result[1::4] = mask_img
    result[2::4] = pred_img
    result[3::4] =  box
    
    result_np = np.array(result)
    result_np = result_np.reshape(image_num, 4)
    result_np = np.vstack((result_np, np.array([accu_plot,sens_plot,spec_plot,f1sc_plot]).reshape(1,4)))

else:
    print "'s' argument is wrong"
    sys.exit(0)

grid = gridplot(result_np.tolist(),
                toolbar_location='right',
                sizing_mode=args.type)



save(grid, filename=None)
show(grid)


