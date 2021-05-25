import tkinter as tk


class ControlPanel(tk.Frame):
    """
    Class containing the logic for a Control Panel for the Dataset Creator.
    """

    def __init__(self, parent, toggleCommand=None, *args, **kwargs):
        # Tkinter Setup
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.gestureNameInputLabel = tk.Label(self, text="Gesture Name:")
        self.gestureNameInputLabel.grid(row=0, column=1)
        self.gestureNameInput = TextField(self)
        self.gestureNameInput.grid(row=0, column=2, padx=5)

        self.recordingToggle = ToggleButton(
            self, "Start Recording", "Stop Recording", command=toggleCommand)
        self.recordingToggle.grid(row=0, column=3, padx=10)

    def isRecording(self) -> bool:
        return self.recordingToggle.active


class ToggleButton(tk.Button):
    """
    Simple Toggle Button logic, if the function
    in `command` is specified, it will be called
    when the button is clicked.
    """

    def __init__(self, parent,  inactiveLabel: str, activeLabel: str, active: bool = False, command=None, *args, **kwargs):
        self.inactiveLabel = inactiveLabel
        self.activeLabel = activeLabel
        self.active = active
        self.command = command

        # Tkinter Setup
        tk.Button.__init__(self, parent, text=self.getLabel(),
                           command=self.toggle, *args, **kwargs)

    def getLabel(self) -> str:
        if self.active:
            return self.activeLabel
        return self.inactiveLabel

    def toggle(self) -> None:
        self.active = not self.active
        self.config(text=self.getLabel())
        if self.command:
            self.command()


class TextField(tk.Entry):
    """
    Text field input.
    """

    def __init__(self, parent, *args, **kwargs):
        tk.Entry.__init__(self, parent, *args, **kwargs)

    def value(self) -> str:
        return self.get()
