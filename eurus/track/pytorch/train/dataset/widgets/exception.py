class EurusMissingDependencyError(Exception):
    r"""
    An exception that a dependency required for the requested functionality
    was not detected.
    """
    def __init__(self, package_name):
        super(EurusMissingDependencyError, self).__init__()
        self.message = "You need to install the '{pname}' package in order " \
                       "to use this functionality.".format(pname=package_name)

    def __str__(self):
        return self.message


class IPythonWidgetsMissingError(EurusMissingDependencyError):
    r"""
    Exception that is thrown when an attempt is made to import ipython widgets,
    but they are not installed or available.
    """
    def __init__(self):
        super(IPythonWidgetsMissingError, self).__init__('ipywidgets')
