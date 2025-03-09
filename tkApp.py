import tkinter as tk
import sys

class NonBlockingTkinterApp:
    def __init__(self, title="Non-Blocking Tkinter GUI", width=300, height=200):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")
        self.widgets = {}

    # *args alows to pass extra arguments to callback 
    def bind_button(self, button_text, callback, *args, row=0, column=0):
        """Bind a button to a callback function."""
        button = tk.Button(self.root, text=button_text, command=lambda: callback(*args))
        button.grid(row=row, column=column)
        self.widgets[button_text] = button

    def bind_key(self, key, callback):
        """Bind a keyboard key to a callback function."""
        self.root.bind(key, lambda event: callback(event))

    def bind_slider(self, min_val, max_val, initial_val, callback=lambda event:None, row=0, column=0):
        """Bind a slider to a callback function."""
        slider = tk.Scale(self.root, from_=min_val, to=max_val, orient="horizontal", command=callback)
        slider.set(initial_val)
        slider.grid(row=row, column=column)
        self.widgets[f"Slider ({min_val}-{max_val})"] = slider
        return slider

    def bind_entry(self, default_value, row=0, column=0):
        """Bind an Entry widget to a callback function."""
        entry = tk.Entry(self.root)
        entry.insert(0, default_value)  # Set default value
        entry.grid(row=row, column=column)
        self.widgets["Entry"] = entry
        return entry

    def bind_entry_with_label(self, label_text, default_value, row=0, column=0):
        """Bind an Entry widget to a callback function."""
        label = tk.Label(self.root, text=label_text)
        entry = tk.Entry(self.root)
        entry.insert(0, default_value)  # Set default value
        label.grid(row=row, column=column)
        entry.grid(row=row, column=column+1)
        self.widgets["Entry"] = entry
        self.widgets[label_text] = label
        return entry
    
    def bind_label(self, label_text, row=0, column=0):
        """Bind an Entry widget to a callback function."""
        label = tk.Label(self.root, text=label_text)
        label.grid(row=row, column=column)
        self.widgets[label_text] = label
        return label



    def update(self):
        """Non-blocking update loop."""
        self.root.update_idletasks()
        self.root.update()


# Usage example
def button_callback():
    print("Button pressed!")

def reset_callback():
    print("Reset pressed!")

def key_callback(event):
    print(f"Key pressed: {event.keysym}")

def quit_callback(event):
    sys.exit()

def slider_callback(value):
    print(f"Slider value: {value}")


if __name__ == "__main__":
    app = NonBlockingTkinterApp()

    # Bind buttons
    app.bind_button("Submit", button_callback, row=0, column=0)
    app.bind_button("Reset", reset_callback, row=1, column=0)

    # Bind keys
    app.bind_key("<Return>", key_callback)  # Bind 'Enter' key
    app.bind_key("<q>", quit_callback)  # Bind 'q' key for quitting

    # Bind slider
    slider = app.bind_slider(0, 100, 50, slider_callback, row=2, column=0)

    # Bind entry
    entry = app.bind_entry("Default Text", row=3, column=0)

    # Non-blocking loop to keep the GUI responsive
    while True:
        app.update()

        # Retrieve slider value
        # print(f"Current slider value: {slider.get()}")

        # Retrieve entry value
        # print(f"Current entry value: {entry.get()}")

        # You can add custom conditions or break the loop if necessary.
        # For example:
