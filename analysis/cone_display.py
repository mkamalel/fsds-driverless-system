from fastslam2 import *
import math

import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
import numpy as np
import pickle
import pandas as pd

cones_df = pd.read_csv("cones_competitionTrack2.csv")

print(cones_df)

x_list = []
y_list = []
color_list = []

for col in cones_df.columns:
  if "location.x" in col:
    x_list.append(cones_df[col][0])
    print("x: ", cones_df[col][0])
  elif "location.y" in col:
    y_list.append(cones_df[col][0])
    print("y: ", cones_df[col][0])
  elif "color" in col:
    color_list.append(cones_df[col][0])
    print("color: ", cones_df[col][0])

processed_cones_df = pd.DataFrame(data={"x":x_list, "y":y_list, "color":color_list})
processed_cones_df.to_csv("processed_cones_competitionTrack2.csv", index=False)

processed_cones_df = pd.read_csv("processed_cones_competitionTrack2.csv")
x_list = processed_cones_df['x'].tolist()
y_list = processed_cones_df['y'].tolist()
color_list = processed_cones_df['color'].tolist()

colors = ["blue", "yellow"]
plt.scatter(x_list, y_list, s=150, c=color_list, cmap=matplotlib.colors.ListedColormap(colors))


plt.show()