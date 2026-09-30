import logging
import os

from smarties4planck.classes import *
from smarties4planck.core_functions import *
from smarties4planck.external_qp_planck import *

_pkg = logging.getLogger("smarties4planck")

if not logging.getLogger().handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.addHandler(handler)


_pkg.setLevel(os.environ.get("smarties4planck_LOGLEVEL", "INFO").upper())
