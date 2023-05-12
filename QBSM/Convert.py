from moviepy.editor import *

filename = "D:/上課資料/IME/實驗室研究/Paper/Coverage Control/Quality based switch mode/Data/Boost_Gazebo.3.29.mp4"
Exportname = "D:/上課資料/IME/實驗室研究/Paper/Coverage Control/Quality based switch mode/Data/Boost_Gazebo.3.29.gif"

video = VideoFileClip(filename).subclip(00,26)
video.write_gif(Exportname, fps = 30)