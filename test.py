from specforge.library.arf_mod import GaussianARFLibrary
from specforge.utils import get_xspec

xspec = get_xspec()
xspec.Xset.chatter = 0
xspec.Xset.abund = 'wilm'
xspec.Fit.query = 'no'
xspec.Fit.statMethod = 'cstat'
xspec.Fit.nIterations = 1000

mu = [0.1,1,2,3,4,5,6,7,8,9,10,11,12]
sigma = [1]
A=[0.08]

# Create the library.
lib = GaussianARFLibrary.create_library("./test",dict(mu=mu,sigma=sigma,A=A),
                                        "../../emma_project/unmodified_arf.arf",
                                        "../../emma_project/center_6102.rmf",
                                         overwrite=True)

lib.generate_library([0.1,5,10,15])

