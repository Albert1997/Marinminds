import numpy as np
import mayavi.mlab as mlab

pc = np.fromfile('output/0001.bin', dtype=np.float32).reshape(-1, 4)

fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
mlab.points3d(
    pc[:, 0],   # x
    pc[:, 1],   # y
    pc[:, 2],   # z
    pc[:, 3],   # Intensity data used for shading
    mode="point", # How to render each point {'point', 'sphere' , 'cube' }
    colormap='spectral',  # 'bone', 'copper',
    # color=(0, 1, 0),     # Used a fixed (r,g,b) color instead of colormap
    scale_factor=100,     # scale of the points
    line_width=10,        # Scale of the line, if any
    figure=fig,
)
# pc[:, 3], # reflectance values
mlab.show()
input()