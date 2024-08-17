"""
Location of data files
======================

Use as ::

    from transport_analysis.data.files import *

"""

__all__ = [
    "MDANALYSIS_LOGO",  # example file of MDAnalysis logo
    "ec_traj_trr",  # ethylene carbonate trajectory in h5md format
    "ec_top",  # ethylene carbonate topology
]

from pkg_resources import resource_filename

MDANALYSIS_LOGO = resource_filename(__name__, "mda.txt")

ec_traj_trr = resource_filename(__name__, "ethylene_carbonate/trajectory.trr")
ec_top = resource_filename(__name__, "ethylene_carbonate/topology.pdb")
