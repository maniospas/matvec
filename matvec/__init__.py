from .matvec import *
import os

matvec_link = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1])+os.sep+"matvec"+os.sep+"py.pyd"
set_lib_file(matvec_link)
set_number_of_threads(os.cpu_count())
