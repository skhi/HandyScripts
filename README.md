# HandyScripts

# Instructions for display_images.py 

Install Bokeh python tool

Install pandas confusion matrix (pandas_ml)

Indicate in the file.txt which pictures you wanna display. The format includes two columns (refer to file.txt): 

   **a.** Number of the image, as they appear in the **prediction** directory from the framework 
  
   **b.** Actual names of the images

To execute the script, type this in the terminal:

  **python display_images.py -f "file.txt" -p  "/path/to/prediction/folder/from/framewokr" -n "name of html output file (for example: frist30images.html)"**

Execute **python display_images.py -h** for more information
