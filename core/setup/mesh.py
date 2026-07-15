
import numpy as np
import matplotlib.pyplot as plt

class Cells:

    def __init__(self, nx, ny):
        self.ids = np.arange(nx*ny).reshape((ny, nx))
        self.owner_x = self.ids[:, :-1]
        self.owner_y = self.ids[:-1, :]
        self.neigh_x = self.ids[:, :-1]
        self.neigh_y = self.ids[:, 1:]

class Mesh:

    def __init__(self, nx, ny, lx, ly, rx, ry):

        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.rx = rx
        self.ry = ry

        self.cells = Cells(nx, ny)



if __name__ == '__main__':

    mesh = Mesh(nx=40, ny=40, lx=2, ly=1, rx=1.1, ry=1.1)

    print(mesh.cells.neigh_x)