"""
Location of data files
======================

Use as ::

    from transport_analysis.data.files import *

"""

__all__ = [
    "MDANALYSIS_LOGO",  # example file of MDAnalysis logo
    "PRM_NCBOX", "TRJ_NCBOX",  # Amber parm7 + nc w/ pos/forces/vels/box
]

from pkg_resources import resource_filename

MDANALYSIS_LOGO = resource_filename(__name__, "mda.txt")
PRM_NCBOX = resource_filename(__name__, "Amber/ace_tip3p.parm7")
TRJ_NCBOX = resource_filename(__name__, "Amber/ace_tip3p.nc")
