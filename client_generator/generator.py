from JSONencoder.JSON_encoder import Generator
from SLAE import SLAE
from surface_area import surface_area as SA

#gen = Generator(SLAE.HomSLAE(3, 0, 3, 0, 0), 2)
#for i in gen.gen():
#    print(i)

#gen2 = Generator(SLAE.SLAEParam(3, 0, 3, 1), 2)
#for i in gen2.gen():
#    print(i)

gen3 = Generator(SA.SurfaceArea1(1, 1))
for i in gen3.gen():
    print(i)