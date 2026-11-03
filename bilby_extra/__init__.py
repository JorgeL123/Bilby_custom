from .utils import *

from .utils.logging import setup_logger
logger = setup_logger()

from .proca import proca_source
from .modGR import modGR_source, st_source
from .eccentric import ecc_source

# Since this reference_time module requires custom lalinference build,
# if the environment is not equipped with that, skipping it.
# During installation, `setup.py` would check for the required function,
# and skipping the relevant module files if not satisfied, we catch
# here the consequent `ImportError`.
try:
    from .reference_time import tref_source
except ImportError:
    logger.warning('Cannot import `reference_time` module, skipping it.')
# , qscan_utils
