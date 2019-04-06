import matlab.engine

engine = matlab.engine.start_matlab()
inputFile = '开关柜.mp4'
outputFile = '输出.avi'
engine.Video2Edges_Canny(inputFile, outputFile, nargout=0)
