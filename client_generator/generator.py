from JSONencoder.JSON_encoder import Generator
from SLAE import SLAE
from surface_area import surface_area as SA

#gen = Generator(SLAE.HomSLAE(3, 0, 3, 0, 0), 2)
#for i in gen.gen():
#    print(i)

#gen2 = Generator(SLAE.SLAEParam(3, 0, 3, 1), 2)
#for i in gen2.gen():
#    print(i)

#gen3 = Generator(SLAE.SLAE(3, 0, 3), 2)

with open("1Площадь.txt", "w+", encoding="UTF-8") as file:
    gen = Generator(SA.SurfaceArea1(1, 1))
    for i in gen.gen():
        file.write(i + ",\n")

with open("2Площадь.txt", "w+", encoding="UTF-8") as file:
    gen = Generator(SA.SurfaceArea2(1, 1))
    for i in gen.gen():
        file.write(i + ",\n")

with open("3Площадь.txt", "w+", encoding="UTF-8") as file:
    gen = Generator(SA.SurfaceArea3(1, 1))
    for i in gen.gen():
        file.write(i + ",\n")

with open("4Площадь.txt", "w+", encoding="UTF-8") as file:
    gen = Generator(SA.SurfaceArea4(1, 1))
    for i in gen.gen():
        file.write(i + ",\n")