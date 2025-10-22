import customtkinter


class MyRadiobuttonFrame(customtkinter.CTkFrame):
    def __init__(self, master, title, values):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.values = values
        self.title = title
        self.radiobuttons = []
        self.variable = customtkinter.StringVar(value="")
        self.title = customtkinter.CTkLabel(master=self, text=title, fg_color="gray30", corner_radius=6)
        self.title.grid(row=0, column=0, padx=20, pady=(10, 0), sticky="ew")

        for i, value in enumerate(self.values):
            radiobutton = customtkinter.CTkRadioButton(master=self, text=value, variable=self.variable, value=value)
            radiobutton.grid(row=i+1, column=0, padx=20, pady=(20, 0), sticky="w")
            self.radiobuttons.append(radiobutton)
    
    def get(self):
        return self.variable.get()
    
    def set(self, value):
        self.variable.set(value)


class CheckboxFrame(customtkinter.CTkFrame):
    def __init__(self, master, values, title ):
        super().__init__(master)
        self.values = values
        self.checkboxes = []
        self.title = title
        self.title = customtkinter.CTkLabel(master=self, text=title, fg_color="gray30", corner_radius=6)
        self.title.grid(row=0, column=0, padx=20, pady=(10, 0), sticky="ew")
        self.grid_columnconfigure(0, weight=1)

        for i, value in enumerate(self.values):
            checkbox = customtkinter.CTkCheckBox(master=self, text=value)
            checkbox.grid(row=i+1, column=0, padx=20, pady=(20, 0), sticky="w")
            self.checkboxes.append(checkbox)
        
    
    def get(self):
        checked_values = []
        for i, checkbox in enumerate(self.checkboxes):
            if checkbox.get():
                checked_values.append(checkbox.cget("text"))
        return checked_values

class MyScrollableCheckboxFrame(customtkinter.CTkScrollableFrame):
    def __init__(self, master, values, title ):
        super().__init__(self, master, label_text=title)
        self.grid_columnconfigure(0, weight=1)
        self.values = values
        self.checkboxes = []

        for i, value in enumerate(self.values):
            checkbox = customtkinter.CTkCheckBox(master=self, text=value)
            checkbox.grid(row=i, column=0, padx=20, pady=(20, 0), sticky="w")
            self.checkboxes.append(checkbox)
    def get(self):
        checked_values = []
        for i, checkbox in enumerate(self.checkboxes):
            if checkbox.get():
                checked_values.append(checkbox.cget("text"))
        return checked_values

class GUI(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Optimizer Visualizer")
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure((0,1), weight=1)

        self.checkbox_frame = CheckboxFrame(master=self, values=["Option 1", "Option 2", "Option 3", "Option 4"], title="Select Options")
        self.checkbox_frame.grid(row=0, column=0, padx=20, pady=(10, 0), sticky="nsew")

        self.checkbox_frame_2 = CheckboxFrame(master=self, values=["Choice A", "Choice B", "Choice C"], title="Select Choices")
        self.checkbox_frame_2.grid(row=0, column=1, padx=20, pady=(10, 0), sticky="nsew")

        self.radiobutton_frame = MyRadiobuttonFrame(master=self, title="Select One", values=["Red", "Green", "Blue"])
        self.radiobutton_frame.grid(row=1, column=0, padx=20, pady=20, sticky="ew", columnspan=2)

        self.scrollabe_checkbox_frame = MyScrollableCheckboxFrame(master=self, values=[f"Item {i}" for i in range(1, 21)], title="Scrollable Items")
        self.scrollabe_checkbox_frame.grid(row=0, column=2, rowspan=3, padx=20, pady=20, sticky="nsew")

        button = customtkinter.CTkButton(master=self, text="Click Me", command=self.button_callback)
        button.grid(row=2, column=0, padx=20, pady=20, sticky= "ew", columnspan=2)

    def button_callback(self):
        print(f"Checked: {self.checkbox_frame.get()}")
        print(f"Checked_2: {self.checkbox_frame_2.get()}")
        print(f"Selected: {self.radiobutton_frame.get()}")
        print(f"Scrollable Checked: {self.scrollabe_checkbox_frame.get()}")

GUI().mainloop()