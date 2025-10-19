import os
import glob
import importlib.util
import inspect
from pathlib import Path


# List to hold all imported surfaces
surface_classes = []

# Dynamically import all surface modules in the current directory and place them into the surfaces list
surface_files = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
for file in surface_files:
    module_name = Path(file).stem
    if module_name != "__init__":
        module = importlib.util.spec_from_file_location(module_name, file)
        loaded_module = importlib.util.module_from_spec(module)
        module.loader.exec_module(loaded_module)
        module_surfaces = [obj for _, obj in inspect.getmembers(loaded_module, inspect.isclass) if issubclass(obj, loaded_module.Surface_Interface) and obj.__module__ == loaded_module.__name__]
        surface_classes.extend( module_surfaces )