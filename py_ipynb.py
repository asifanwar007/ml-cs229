"""Create a notebook containing code from a script.
Run as:  python make_nb.py my_script.py
"""
import sys
import argparse

import nbformat
from nbformat.v4 import new_notebook, new_code_cell

nb = new_notebook()
with open(sys.argv[1]) as f:
    code = f.read()
# parser = argparse.ArgumentParser()
# parser.add_argument('file_name',nargs='?', type=str, default="nothing", 'w_file',nargs='?', type=str, default="nothing",
#                     help='please give file name as arg')
# args = parser.parse_args()
nb.cells.append(new_code_cell(code))

pos = len(sys.argv)
if pos == 2:
	print("You haven't entered the file name")
	print("Press 'Y' to keep the file name same or any key to no:", end=" ")
	inp = input()
	if(inp.lower() == 'y'):
		nbformat.write(nb, sys.argv[1]+'.ipynb')
	else:
		print("Enter new file Name: ", end=" ")
		inp = input()
		if inp is '':
			print("You haven't entered the file name")
			exit()
		else:
			nbformat.write(nb, inp+'.ipynb')


else:
	nbformat.write(nb, sys.argv[2]+'.ipynb')


