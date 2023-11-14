import tkinter as tk
import datetime
import pandas as pd


class DfViewer(tk.Frame):

    def show(self, entry: pd.DataFrame, processed=False):
        if type(entry) == pd.DataFrame:
            if not processed:
                self.df: pd.DataFrame = entry
                self.df_processed = entry
                self.update_lbs()

            newline = '\n'
            header = [f'{i[-7:]}' for i in entry.columns]
            self.text = entry.to_string(
                float_format=lambda x: '{:.3f}'.format(x), header=header)

        self.tb.delete('1.0', tk.END)
        self.tb.insert(tk.END, self.text)
        self.lift()

    def update_lbs(self):
        self.lb_sort.delete(0, tk.END)
        for col in self.df.columns:
            self.lb_sort.insert(tk.END, col)

        self.lb_select.delete(0, tk.END)
        for col in self.df.columns:
            self.lb_select.insert(tk.END, col)

    def sort_cols(self, _event=False):
        col = self.lb_sort.get(self.lb_sort.curselection()[0])
        self.df_processed = self.df_processed.sort_values(col)
        self.show(self.df_processed, processed=True)

    def filter_columns(self):
        cols = []
        for ind in self.lb_select.curselection():
            cols.append(self.lb_select.get(ind))
        if cols:
            self.df_processed = self.df[cols]
        else:
            self.df_processed = self.df
        self.show(self.df_processed, processed=True)

    def exit(self):
        self.master.destroy()

    def __init__(self, _master):

        super().__init__(_master)
        self.master = _master
        self.grid()

        self.df = None
        self.df_processed = None
        self.text = None

        self.sb_sort = tk.Scrollbar(self)
        self.sb_sort.grid(row=1, column=2, sticky=tk.N + tk.S)

        self.lb_sort = tk.Listbox(self, height=20, width=20,
                                  font=('Consolas', 10),
                                  yscrollcommand=self.sb_sort.set,
                                  exportselection=False)
        self.lb_sort.grid(row=1, column=1)
        self.sb_sort.config(command=self.lb_sort.yview)
        self.lb_sort.bind('<<ListboxSelect>>', self.sort_cols)

        self.sb_select = tk.Scrollbar(self)
        self.sb_select.grid(row=1, column=4, sticky=tk.N + tk.S)

        self.lb_select = tk.Listbox(self, height=20, width=20,
                                    font=('Consolas', 10),
                                    yscrollcommand=self.sb_select.set,
                                    selectmode=tk.EXTENDED,
                                    exportselection=False)
        self.lb_select.grid(row=1, column=3)
        self.sb_select.config(command=self.lb_select.yview)


        self.tb = tk.Text(self, wrap=tk.NONE, height=40, width=140)
        self.tb.configure(font=('Consolas', 10))
        self.tb.grid(row=1, column=5)

        self.sb_tb_x = tk.Scrollbar(self, orient=tk.HORIZONTAL,
                                    command=self.tb.xview)
        self.sb_tb_x.grid(row=2, column=5, sticky='ew')
        self.sb_tb_y = tk.Scrollbar(self, command=self.tb.yview)
        self.sb_tb_y.grid(row=1, column=6, sticky=tk.S + tk.N)
        self.tb.configure(xscrollcommand=self.sb_tb_x.set)
        self.tb.configure(yscrollcommand=self.sb_tb_y.set)

        self.bt_filter_columns = tk.Button(self, text='Filter',
                                           command=self.filter_columns)
        self.bt_filter_columns.grid(row=2, column=1)

        self.bt_exit = tk.Button(self, text='Exit',
                                           command=self.exit)
        self.bt_exit.grid(row=3, column=1)

        self.l_message = tk.Label(self, text='Messages', wraplength=450)
        self.l_message.grid(row=3, column=5)

def create_dfviewer(entry=None, title=None, message=None):
    window_dfviewer_top = tk.Toplevel()
    window_dfviewer_top.geometry('1350x800+5+5')

    window_dfviewer = DfViewer(window_dfviewer_top)

    window_dfviewer_top.lift()

    if entry is not None:
        window_dfviewer.show(entry)

    if title:
        window_dfviewer_top.title(title)
    else:
        time = datetime.datetime.now().isoformat()[0:19]
        window_dfviewer_top.title(f'Df viewer [{time}]')

    if message:
        window_dfviewer.l_message['text'] = message

    return window_dfviewer
