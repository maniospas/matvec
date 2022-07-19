from .matvec import *
import warnings

matvec_link = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1])+os.sep+"matvec"+os.sep+"py.pyd"
if os.path.exists(matvec_link):
    load_matvec(matvec_link)
else:
    warnings.warn("Failed to load built matvec from location: "+matvec_link)
