from PlanckConv.classes import *
from PlanckConv.core_functions import *
from PlanckConv.external_qp_planck import *

import logging, os

_pkg = logging.getLogger("PlanckConv")

if not logging.getLogger().handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))
    root = logging.getLogger()
    root.addHandler(handler)


_pkg.setLevel(os.environ.get("PLANCKCONV_LOGLEVEL", "INFO").upper())
