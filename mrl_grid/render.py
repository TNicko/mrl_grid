# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle
# from matplotlib.colors import ListedColormap
# import numpy as np
# import time

# # Map of color names to Hex values
# COLORS = {
#     'red'         : '#e8074a',
#     'light_red'   : '#e68aa5',
#     'grey'        : '#646464',
#     'black'       : '#000000',
#     'white'       : '#ffffff'
# }
# COLOR_NAMES = sorted(list(COLORS.keys()))

# class Grid:
#     """
#     Represent a grid and operations on it
#     """

#     def __init__(self, width, height):

#         self.width = width
#         self.height = height


#     def render(current_pos, width, height):

#         img = 

#         # Add agent to grid at current position
#         x, y = current_pos
#         circle = Circle((y, x), radius=0.3, color=COLORS['red'])
#         ax.add_patch(circle)

#         # Create grid layout
#         for i in range(width):
#             ax.axhline(i - 0.5, color='black', linewidth=1)
#         for j in range(height):
#             ax.axvline(j - 0.5, color='black', linewidth=1)