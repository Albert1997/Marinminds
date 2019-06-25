from pyntcloud import PyntCloud

pc = PyntCloud.from_file("datasets/2018-06-13-13-45-05_Velodyne-VLP-16-Data_Frame_0081.csv", sep=",", skiprows=1, usecols=[3,4,5], names=["x", "y", "z"])
pc.to_file("output/0001.ply")