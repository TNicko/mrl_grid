
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle

# Map of color names to Hex values
COLORS = {
    'red'         : '#e8074a',
    'light_red'   : '#e68aa5',
    'grey'        : '#646464',
    'black'       : '#000000',
    'white'       : '#ffffff'
}
COLOR_NAMES = sorted(list(COLORS.keys()))

class Window:
    """
    Window to draw a gridworld instance using Matplotlib
    """

    def __init__(self, title, width, height, fps):

        self.fig, self.ax = plt.subplots()

        self.width = width
        self.height = height
        self.fps = fps

        # Show the env name in the window title
        self.fig.canvas.manager.set_window_title(title)
        self.cmap = ListedColormap([COLORS['white'], COLORS['light_red']])

    def render(self, grid, current_pos):
        """
        Render image or update the image being shown
        """
        self.ax.clear()

        self.ax.imshow(grid, cmap=self.cmap, vmin=0, vmax=1)

        # Remove ticks on axis
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Create grid layout
        for i in range(self.width):
            self.ax.axhline(i - 0.5, color='black', linewidth=1)
        for j in range(self.height):
            self.ax.axvline(j - 0.5, color='black', linewidth=1)

        # Add agent to grid at current position
        x, y = current_pos
        circle = Circle((y, x), radius=0.3, color=COLORS['red'])
        self.ax.add_patch(circle)

        # Request the window be redrawn
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # self.fig.canvas.draw()  # Redraw the figure
        plt.pause(1/self.fps)

    def show(self):
        plt.ion()

    def close(self):
        """
        Close the window
        """
        plt.close()


