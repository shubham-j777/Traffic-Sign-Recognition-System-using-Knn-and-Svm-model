import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
import joblib

# Load the trained KNN/SVM model
model_file_path = "D:/college files/DS course/trafic recog project code/GTSRB/traffic_sign_knn_model.pkl"
knn_model = joblib.load(model_file_path)

sign_names = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing veh over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End speed + passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing veh > 3.5 tons'
}


class TrafficSignRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TRAFFIC SIGN RECOGNITION")
        self.root.geometry("1128x634")

        # Load and display the background image
        self.background_image = Image.open("D:/college files/DS course/trafic recog project code/GTSRB/GUI_Background.jpg")
        self.background_image = self.background_image.resize((1128, 634), Image.LANCZOS)
        self.background_image_tk = ImageTk.PhotoImage(self.background_image)
    
        self.background_label = tk.Label(self.root, image=self.background_image_tk)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        self.create_widgets()




    def create_widgets(self):
        # Other widgets within the recognition frame
        frame = tk.Frame(self.root, relief=tk.SOLID, borderwidth=2)
        self.label_title = tk.Label(self.root, text="TRAFFIC SIGN RECOGNITION", font=("Helvetica", 30),bg="#add8e6",relief=tk.SOLID, borderwidth=2)
        self.label_title.pack(pady=20)
        frame.pack()
        
        # Create a frame for buttons and place them on the same line
        button_frame = tk.Frame(self.root, bg="grey")
        button_frame.pack()

        self.button_upload = tk.Button(button_frame, text="Upload Image", command=self.select_image, font=("Helvetica", 14))
        self.button_upload.pack(side="left", padx=1)
        
        self.button_show_all = tk.Button(button_frame, text="Show All Images", command=self.show_all_images, font=("Helvetica", 14))
        self.button_show_all.pack(side="left", padx=1)
        
        self.label_image = tk.Label(self.root,bg="black")
        self.label_image.pack(pady=10)
        
        self.label_result = tk.Label(self.root, text="", font=("Helvetica", 16),bg="white")
        self.label_result.pack(pady=10)
        
        
        
    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if file_path:
            self.predict_traffic_sign(file_path)
    
    
    def predict_traffic_sign(self, image_path): #Main Prediction Function
        image = Image.open(image_path)
        image.thumbnail((300, 300))
        resized_image = image.resize((250, 250), Image.LANCZOS)  # Resize the image to a fixed size
        image_tk = ImageTk.PhotoImage(resized_image)
        self.label_image.config(image=image_tk)
        self.label_image.image = image_tk
        
        predicted_label, predicted_sign_name = self.predict1(image_path) 
        
        result_text = f"Predicted Class: {predicted_label+1}\nPredicted Sign: {predicted_sign_name}"
        self.label_result.config(text=result_text)



    def predict1(self, image_path): #Prediction FUnction
        image = Image.open(image_path)
        image_gray = rgb2gray(np.array(image))
        image_resized = resize(image_gray, (32, 32))
        # Extract HOG features
        features = hog(image_resized, block_norm='L2-Hys', pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        # Predict using the trained KNN model
        predicted_label = knn_model.predict([features])[0]
        predicted_sign_name = sign_names.get(predicted_label, "Unknown")
        
        return predicted_label, predicted_sign_name


        
    def show_all_images(self):
        image_path = "D:/college files/DS course/trafic recog project code/GTSRB/MetaImage1080.jpg"  # Provide the actual path to the image
        self.display_image_in_new_window(image_path)    
        
    def display_image_in_new_window(self, image_path):
        new_window = tk.Toplevel(self.root)
        new_window.title("Showing All Images")
        
        image = Image.open(image_path)
        resized_image = image.resize((1128, 634), Image.LANCZOS)  # Resize the image to  fit the windows
        self.image_tk = ImageTk.PhotoImage(resized_image)
        
        label = tk.Label(new_window, image=self.image_tk,bg='black')
        label.pack(padx=10, pady=10)
            
            
if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignRecognitionApp(root)
    root.mainloop()
