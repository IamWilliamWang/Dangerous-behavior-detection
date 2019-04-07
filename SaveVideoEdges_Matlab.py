import matlab
import matlab.engine
import numpy as np
import imageio

matengine = matlab.engine.start_matlab()
edges = matengine.getEdgesFromVideo(nargout=1)
outMat = np.asarray(matengine.getEdgesFromVideo('开关柜.mp4', nargout=1))
imageio.mimsave('output.mp4', outMat)
