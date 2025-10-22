import os
import glob
import importlib.util
import inspect
from pathlib import Path


# TODO: Check if new optimizers are imported dynaiqmically when a new file is added

# List to hold all imported optimizer
optimizer_classes = []

# Dynamically import all optimizer modules in the current directory and place them into the optimizers list
optimizer_files = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
for file in optimizer_files:
    module_name = Path(file).stem
    if module_name != "__init__":
        module = importlib.util.spec_from_file_location(module_name, file)
        loaded_module = importlib.util.module_from_spec(module)
        module.loader.exec_module(loaded_module)
        module_optimizers = [obj for _, obj in inspect.getmembers(loaded_module, inspect.isclass) if issubclass(obj, loaded_module.Optimizer_Interface) and obj.__module__ == loaded_module.__name__]
        optimizer_classes.extend( module_optimizers )