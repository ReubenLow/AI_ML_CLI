import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os

# Main GUI class
class PipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Pipeline GUI")
        self.root.geometry("400x300")
        
        # Directory selection buttons
        self.create_button("Select Cleaned Data Folder", self.select_cleaned_data_folder)
        self.create_button("Select Model Output Folder", self.select_model_output_folder)
        
        # File selection for EDA
        self.create_button("Select Data for EDA", self.select_eda_file)
        
        # Run buttons for each stage
        self.create_button("Run Data Cleaning", self.run_data_cleaning)
        self.create_button("Run EDA", self.run_eda)
        self.create_button("Run Model Training", self.run_model_training)
        self.create_button("Run Prediction", self.run_prediction)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Status: Waiting")
        self.status_label.pack(pady=10)

    def create_button(self, text, command):
        button = tk.Button(self.root, text=text, command=command, width=30)
        button.pack(pady=5)

    # Directory and file selection functions
    def select_cleaned_data_folder(self):
        self.cleaned_data_folder = filedialog.askdirectory()
        self.status_label.config(text=f"Selected Cleaned Data: {self.cleaned_data_folder}")

    def select_model_output_folder(self):
        self.model_output_folder = filedialog.askdirectory()
        self.status_label.config(text=f"Selected Model Output: {self.model_output_folder}")

    def select_eda_file(self):
        self.eda_file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        self.status_label.config(text=f"Selected EDA File: {self.eda_file}")

    # Run functions for each stage
    def run_data_cleaning(self):
        self.run_script("clean_data.py", "cleaned_data_folder")

    def run_eda(self):
        self.run_script("perform_eda.py", "eda_file")

    def run_model_training(self):
        self.run_script("model_training.py", "model_output_folder")

    def run_prediction(self):
        self.run_script("perform_prediction.py", "model_output_folder")

    def run_script(self, script_name, folder_name):
        try:
            subprocess.run(["python", script_name, "--input_folder", getattr(self, folder_name)], check=True)
            self.status_label.config(text=f"{script_name} completed successfully")
        except subprocess.CalledProcessError:
            messagebox.showerror("Error", f"Failed to run {script_name}")
            self.status_label.config(text=f"Error: {script_name} failed")

# Initialize the GUI
root = tk.Tk()
app = PipelineGUI(root)
root.mainloop()
