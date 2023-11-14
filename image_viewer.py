import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.image as mpimg


class WindowImages(tk.Frame):

    def __init__(self, parent, width_image, width_count, app_main):
        tk.Frame.__init__(self, parent)
        self.width_image = width_image
        self.width_count = width_count
        self.canvas = tk.Canvas(self, borderwidth=0)
        self.frame = tk.Frame(self.canvas)
        self.sb_y = tk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.sb_y.set)
        self.sb_y.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.canvas.create_window((4, 4), window=self.frame, anchor='nw',
                                  tags=self.frame)
        self.frame.bind("<Configure>", self.onframeconfigure)

        self.frame.bind('<Enter>', self.bind_to_mousewheel)
        self.frame.bind('<Leave>', self.unbind_to_mousewheel)

        self.labels_img = []
        self.labels_name = []
        self.meass_toplot = []
        self.images_raw = []
        self.app_main = app_main

    def bind_to_mousewheel(self, event):
        pass

    def unbind_to_mousewheel(self, event):
        pass

    def on_mousewheel(self, event):
        if event.num == 5:
            direction = 1
        else:
            direction = -1
        self.canvas.yview_scroll(direction, 'units')

    def onframeconfigure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def clear(self):
        for label_name in self.labels_name:
            label_name.destroy()
        for label_img in self.labels_img:
            label_img.destroy()
        self.meass_toplot = []
        self.images_raw = []

    def populate(self, paths=None, images_raw=None, labels=None,
                 click_function=None, meass_toplot=None):

        self.clear()

        if meass_toplot:
            self.meass_toplot = meass_toplot

        if not paths and not images_raw:
            return

        row = 1
        column = 1

        images_tk = []
        self.labels_img = []
        self.labels_name = []
        self.labels_name_strings = []

        if paths and not images_raw:

            images_raw = []

            for path in paths:
                images_raw.append(Image.open(path))

        self.images_raw = images_raw

        for i, image_raw in enumerate(images_raw):

            image_scaled = image_raw.resize(
                (self.width_image,
                 int(image_raw.height * (self.width_image / image_raw.width))),
                Image.LANCZOS)
            image_tk = ImageTk.PhotoImage(image_scaled)

            images_tk.append(image_tk)

            if paths and not labels:
                label_string = 'N/A'
            elif labels:
                label_string = labels[i]
            else:
                label_string = 'N/A'

            label_name = tk.Label(
                self.frame, text=label_string)
            label_name.grid(row=row, column=column, pady=10)
            self.labels_name_strings.append(label_string)

            label_img = tk.Label(self.frame, image=image_tk, name=str(i))
            label_img.image = image_tk
            label_img.bind(
                '<Button-1>', lambda event: show_image_picked(event, self))
            label_img.grid(row=row+1, column=column)

            self.labels_img.append(label_img)

            if (i+1) % self.width_count == 0:
                row += 2
                column = 1
                pass
            else:
                column += 1


def show_image_picked(event, window_images):

    print(f'Showing picked image...')

    col = event.widget.grid_info()['column']
    row = int(event.widget.grid_info()['row'] / 2)

    i_image = (row - 1) * window_images.width_count + (col) - 1


    meas_toplot = window_images.meass_toplot[i_image]
    label_name = window_images.labels_name_strings[i_image]

    fig_image = window_images.app_main.figures.get_pick(clearax=True, connect_click=True)
    fig_image.clear_dist()

    fig_image.title = label_name
    fig_image.ax_p.set_title(fig_image.title)

    if 'nparray' not in meas_toplot.data['modes']['image']:
        print(f'Loading image file: {meas_toplot.data["file"]["path"]}')
        img = mpimg.imread(meas_toplot.data['file']['path'])
        meas_toplot.data['modes']['image']['nparray'] = img

    if 'm_pixel' in meas_toplot.data['modes']['image']:
        m_pixel = meas_toplot.data['modes']['image']['m_pixel']
        size_x = meas_toplot.data['modes']['image']['size_x']
        size_y = meas_toplot.data['modes']['image']['size_y']

        fig_image.ax_p.imshow(
            meas_toplot.data['modes']['image']['nparray'],
            extent=[0, size_x * m_pixel, 0, size_y * m_pixel])

        fig_image.ax_p.set_xlabel('m')
        fig_image.ax_p.set_ylabel('m')

    else:
        fig_image.ax_p.imshow(meas_toplot.data['modes']['image']['nparray'])
        fig_image.ax_p.set_xlabel('pixels')
        fig_image.ax_p.set_ylabel('pixels')

    fig_image.fig.show()
