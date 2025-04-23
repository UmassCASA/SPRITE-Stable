import logging
import pkgutil
import importlib


class AbstractClassScanner:
    def __init__(self, package, base_class):
        """
        Initialize the scanner.

        :param package: The package to scan (can be an already imported package object).
        :param base_class: The abstract base class whose subclasses will be searched (class object).
        """
        self.package = package
        self.base_class = base_class
        self._classes = None

    def import_submodules(self):
        """
        Traverse and import all submodules and subpackages under the given package.
        """
        package_name = self.package.__name__
        # package.__path__ is a list containing the package's search paths.
        for _loader, name, _is_pkg in pkgutil.walk_packages(self.package.__path__, package_name + "."):
            importlib.import_module(name)

    def get_all_subclasses(self):
        """
        Recursively retrieve all subclasses of the base_class.

        :return: A set containing all subclasses.
        """
        # Ensure that all modules in the package are imported; otherwise, some subclasses may be missed.
        self.import_submodules()

        def _recursive_subclasses(cls):
            subclasses = set(cls.__subclasses__())
            for subclass in cls.__subclasses__():
                subclasses |= _recursive_subclasses(subclass)
            return subclasses

        self._classes = _recursive_subclasses(self.base_class)
        return self._classes

    def filter_classes_by_name(self, allowed_names=None, contains=None, exclude=False):
        """
        Filter the scanned classes by their class names, and return a dictionary of {class name: class}.

        :param allowed_names: A list specifying allowed class names (exact match).
            If provided, only classes whose names are in this list will be returned.
        :param contains: A string or a list of strings.
            Only classes whose names contain the string or any of the strings in the list will be returned;
            if allowed_names is provided, this parameter is ignored.
        :return: A dictionary of {class name: class}.
        """
        all_classes = {
            cls for cls in self.get_all_subclasses() if cls.__module__.startswith(self.package.__name__ + ".")
        }

        if allowed_names is not None:
            filtered = {cls.__name__: cls for cls in all_classes if (cls.__name__ in allowed_names) ^ exclude}
        elif contains is not None:
            if isinstance(contains, str):  # If it is a single string.
                filtered = {cls.__name__: cls for cls in all_classes if (contains in cls.__name__) ^ exclude}
            elif isinstance(contains, list):  # If it is a list of strings.
                filtered = {
                    cls.__name__: cls for cls in all_classes if (any(sub in cls.__name__ for sub in contains)) ^ exclude
                }
            else:
                raise ValueError("The `contains` parameter must be a string or a list of strings.")
        else:
            # If no filter condition is provided, return all scanned classes.
            filtered = {cls.__name__: cls for cls in all_classes}
        return filtered

    def filter_allow_then_contains(self, class_name_list=None, suffix_list=None, exclude=False):
        if suffix_list is None:
            suffix_list = [""]

        all_classes = {
            cls for cls in self.get_all_subclasses() if cls.__module__.startswith(self.package.__name__ + ".")
        }

        filtered = {}
        if class_name_list is not None:
            for cls_name in class_name_list:
                matched_class_dict = self.filter_classes_by_name(
                    allowed_names=[cls_name + suffix for suffix in suffix_list], exclude=exclude
                )
                if 0 == len(matched_class_dict):
                    matched_class_dict = self.filter_classes_by_name(contains=cls_name, exclude=exclude)

                filtered.update(matched_class_dict)
        else:
            # If no filter condition is provided, return all scanned classes.
            filtered = {cls.__name__: cls for cls in all_classes}
        return filtered

    def run_all(self, method_name="run"):
        """
        Instantiate each scanned class and call the specified method.

        :param method_name: The name of the method to call, defaults to 'run'.
        """
        all_classes = self.get_all_subclasses()
        for cls in all_classes:
            instance = cls()
            method = getattr(instance, method_name, None)
            if callable(method):
                logging.info(f"Running {cls.__name__}.{method_name}()")
                method()
            else:
                raise AttributeError(f"Class {cls.__name__} does not have method {method_name}")


def import_by_string(dotted_path):
    """
    Dynamically import a module or retrieve an attribute from a module based on a string path.
    For example: 'my_module.MyABC' will import the my_module module and return its MyABC attribute.
    """
    try:
        module_path, attr_name = dotted_path.rsplit(".", 1)
    except ValueError:
        raise ImportError(f"{dotted_path} is not a valid module path") from None
    module = importlib.import_module(module_path)
    try:
        attr = getattr(module, attr_name)
    except AttributeError:
        raise ImportError(f"Module {module_path} does not have attribute {attr_name}") from None
    return attr


# Example usage
if __name__ == "__main__":
    # Assume we want to dynamically pass in a package and an abstract class using string representations:
    package_str = "model_compare.experiments.forecast_methods.Impl"  # Target package name.
    base_class_str = (
        "model_compare.experiments.forecast_methods.Interface."
        "ForecastMethodInterface.ForecastMethodInterface"
    )  # Full path of the abstract base class.

    # Dynamically import the package (note: the package must be in PYTHONPATH).
    target_package = importlib.import_module(package_str)
    # Dynamically import the abstract base class.
    target_base_class = import_by_string(base_class_str)

    # Initialize the scanner with the dynamically imported package and abstract base class.
    scanner = AbstractClassScanner(target_package, target_base_class)

    # Filter classes by name. For example, you may filter to only include certain class names:
    # filtered_classes = scanner.filter_classes_by_name(
    # allowed_names=['DGMRForecast', 'PystepsLindaDeterministicForecast']
    # )
    # filtered_classes = scanner.filter_classes_by_name(
    # contains=[
    #     Constants.PYSTEPS_METHODS_NAME_KEYWORD.value,
    #     Constants.NOWCASTNET_METHODS_NAME_KEYWORD.value
    # ]
    # )
    # filtered_classes = scanner.filter_classes_by_name(None)
    # filtered_classes = scanner.filter_classes_by_name(allowed_names=["Nothing"])
    filtered_classes = scanner.filter_allow_then_contains(
        class_name_list=["UNet"],
        suffix_list=[
            "Forecast",
        ],
    )
    print("Filtered classes:")
    for name, cls in filtered_classes.items():
        print(f"{name}: {cls}")

    # Run the run() method for each filtered class.
    for name, cls in filtered_classes.items():
        # instance = cls()
        print(f"Running {name}.run() ... father: {cls.__bases__[0].__name__}")
        # instance.run()
