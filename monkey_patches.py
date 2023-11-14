""" Add GaAsBi and GaAsN to xrayutilities predefined materials.
(compressive and tensile)
Requires adding GaBi and GaN (zincblende structure).
"""

import xrayutilities


from xrayutilities.materials.material import (Crystal, CubicAlloy, CubicElasticTensor,
                       HexagonalElasticTensor, WZTensorFromCub)
from xrayutilities.materials.spacegrouplattice import SGLattice
from xrayutilities.materials.predefined_materials import GaAs
from xrayutilities.materials import elements as e

GaBi = Crystal("GaBi", SGLattice(216, 6.33, atoms=[e.Ga, e.Bi],
                                 pos=['4a', '4c']),
               CubicElasticTensor(11.9e+10, 5.34e+10, 5.96e+10),
               thetaDebye=360)
GaNzb = Crystal("GaNzb", SGLattice(216, 4.5, atoms=[e.Ga, e.N],
                                 pos=['4a', '4c']),
               CubicElasticTensor(11.9e+10, 5.34e+10, 5.96e+10),
               thetaDebye=360)

xrayutilities.materials.predefined_materials.GaBi = GaBi
xrayutilities.materials.predefined_materials.GaNwz = GaNzb

class GaAsBi(CubicAlloy):

    def __init__(self, x):
        """
        GaAsBi cubic compound
        """
        super().__init__(GaAs, GaBi, x)

class GaAsN(CubicAlloy):

    def __init__(self, x):
        """
        GaAsN cubic compound
        """
        super().__init__(GaAs, GaNzb, x)


xrayutilities.materials.predefined_materials.GaAsBi = GaAsBi
xrayutilities.materials.predefined_materials.GaAsN = GaAsN

xrayutilities.materials.GaAsBi = GaAsBi
xrayutilities.materials.GaAsN = GaAsN