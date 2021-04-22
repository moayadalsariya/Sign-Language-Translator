# 
# ======================================================
#   import libs
# ======================================================
# 
# import the lib
from tkinter import *
from PIL import ImageTk,Image
import tkinter.font as tkFont
from tkinter import filedialog
# import nesscary libary 
# openCV lib
import cv2 as cv 
# matplotlib lib
# numpy lib
import numpy as np
# import os
import os
import tensorflow as tf
from tensorflow import keras
from tkinter import messagebox
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
import shutil 
import webbrowser as wb
root = Tk()
# ini root windows 700x550 pixels
root.title("ASL")
list_nums = []
obj2 = {}
if os.path.exists("./datasets"):
    shutil.rmtree("./datasets")

def submit_train_data(data):
    global list_nums
    list_nums.append(data)
    create_dir()
    obj2[data] = 0
    video = cv.VideoCapture(0)
    while(True):
        isTrue, frame = video.read()
#       flip the image
        frame = cv.flip(frame, 1)
#       draw rect at top right corner of screen
        cv.rectangle(frame, (0,0), (frame.shape[1]//2 - 50, frame.shape[0]//2 - 50), (0,255,0), thickness = 1)
#       Write text
        cv.putText(frame, 'Mode:collecting images', (0,230), cv.FONT_HERSHEY_SIMPLEX , 1.0, (0,255,0), 2)
        cv.putText(frame, 'Count {} : {}'.format(data,obj2[data]), (0,260), cv.FONT_HERSHEY_SIMPLEX , 0.8, (255,0,0), 2) 
#       show the image
        cv.imshow('frame',frame)
        k = cv.waitKey(33)
        if k == 27:    # Esc key to stop
            break
        elif k == -1:  # normally -1 returned,so don't print it
            continue
        elif k == 32:
            print(obj2)
            cropped_img = frame[1:(frame.shape[0]//2) - 50,1:(frame.shape[1]//2) - 50]
            gray = cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY)
            blur = cv.medianBlur(gray, 7)
            th3 = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv.THRESH_BINARY_INV,11,2)
        #   resizing the image
            resized = cv.resize(th3, (100,100), interpolation=cv.INTER_AREA)
        #   save the crop images to target folder
            cv.imwrite('./datasets/{}/{}.jpg'.format(data,str(obj2[data])), resized)
        #   increase the counter
            obj2[data] += 1 
        if(obj2[data] == 100):
            messagebox.showinfo("collecting images", "You have reach 100 images data, now train the model")
            
    video.release()
    cv.destroyAllWindows()
def create_dataset(img_folder):
    img_data_array=[]
    class_name=[]
   
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path= os.path.join(img_folder, dir1,  file)
            image= cv.imread(image_path,0)
            image=np.array(image)
            image = image.astype('float32')
            image /= 255 
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name

obj3 = {}
def train_model_func():
    if not os.path.exists("./datasets"):
        messagebox.showerror("Please create datasets", 'Please register new sign to train model')
    global list_nums
    img_data, class_name =create_dataset('./datasets')
    img_data = np.array(img_data)
    class_name = np.array(class_name)
    global obj3
    for i in list_nums:
        obj3[i] = list_nums.index(i)
    class_name = np.array([obj3[x] for x in class_name])
    img_data = img_data.reshape((len(list_nums)*100, 100, 100,1))
    X_train, X_test, y_train, y_test = train_test_split(img_data, class_name, test_size=0.3, random_state=101)
    model = models.Sequential([
    layers.Conv2D(64, 3, padding='same', activation='relu', input_shape = (100,100,1)),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(list_nums))
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
             
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test,y_test))
    # win2 = Toplevel()
    # win2.title('Second windows')
    # bb = Button(win2, text="Create new window")
    # bb.pack()
    acc = model.evaluate(X_test,y_test)
    model_filename = filedialog.asksaveasfilename(initialdir="./", title="Save A File", filetypes=(("h5 files", "*.h5"),("all files", "*.*")))
    model.save(model_filename)


def new_windows():
    top = Toplevel()
    top.title('Add new gesture')
    fontStyle2 = tkFont.Font(family="Lucida Sign", size=15)
    title_ASL2 = Label(top, text = 'Add new Sign', font = fontStyle2)
    new_entrry = Entry(top, text = "Please enter new sign")
    button_sign = Button(top, text = "Submit", command = lambda: submit_train_data(new_entrry.get()))
    title_ASL2.grid(row = 0, column = 0, columnspan = 1)
    new_entrry.grid(row = 1, column = 0)
    button_sign.grid(row = 1, column = 1)
def create_dir():
    global list_nums
    if not os.path.exists("./datasets"):
        os.makedirs("./datasets")
    for i in list_nums:
            if not os.path.exists(f"./datasets/{i}"):
                os.makedirs(f"./datasets/{i}")   


    

    

def process_img(frame):
        cropped_img = frame[1:(frame.shape[0]//2) - 50,1:(frame.shape[1]//2) - 50]
        gray = cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY)
        blur = cv.medianBlur(gray, 7)
        th3 = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY_INV,11,2)
#   resizing the image
        resized = cv.resize(th3, (100,100), interpolation=cv.INTER_AREA)
        return resized
obj = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5', 6:'6', 7:'7', 8:'8', 9:'9',10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',30:'U',31:'V',32:'W',33:'X',34:'Y',35:'Z',36:' '}

def prediction_clicked():
    global obj
    video = cv.VideoCapture(0)
    root.filename = filedialog.askopenfilename(initialdir="./", title="Select A File", filetypes=(("h5 files", "*.h5"),("all files", "*.*")))
    last_path = os.path.basename(os.path.normpath(root.filename))
    if(last_path != 'model_ASL.h5'):
        obj = dict(zip(obj3.values(), obj3.keys()))
    loaded_model = keras.models.load_model(root.filename)
    while(True):
    #     read the video images
        isTrue, frame = video.read()
    #   flip the image
        frame = cv.flip(frame, 1)
    #   draw rect at top right corner of screen
        cv.rectangle(frame, (0,0), (frame.shape[1]//2 - 50, frame.shape[0]//2 - 50), (0,255,0), thickness = 1)
    #   Write text
        cv.putText(frame, 'Mode:predict model', (0,230), cv.FONT_HERSHEY_SIMPLEX , 1.0, (0,255,0), 2)
    #   process img
        process_images = process_img(frame)
    #   predit image 
        text = np.argmax(loaded_model.predict(process_images.reshape(-1,100,100,1)), axis=-1)
                    
        cv.putText(frame, 'Prediction: {}'.format(obj[text[0]]), (0,290), cv.FONT_HERSHEY_SIMPLEX , 1.0, (255,0,0), 2)
        cv.imshow('frame',frame)
        cv.imshow('Threshold image',process_img(frame))
        k = cv.waitKey(33)
        if k == 27:    # Esc key to stop
            break
    video.release()
    cv.destroyAllWindows()


# 
# ======================================================
#   init widgets
# ======================================================
# 
fontStyle = tkFont.Font(family="Lucida Grande", size=30)
fontStyle2 = tkFont.Font(family="Lucida Grande", size=12)

title_ASL = Label(root, text = 'Sign lanaguge trasnaltor', font = fontStyle)
subtitle = Label(root, text = 'SLT is tool that uses computer vison to track a users gestures', font = fontStyle2)
subtitle2 = Label(root, text = 'And then used a learned model to identify  the SLT character most ', font = fontStyle2)
subtitle3 = Label(root, text = 'correlated to that gesture.', font = fontStyle2)

register_sign = Button(root, text = 'Register new sign', padx = 10, pady = 10, command = new_windows)

train_model = Button(root, text = 'Train new model',padx = 10, pady = 10, command = train_model_func)

make_prediction = Button(root, text = 'Make prediction',padx = 10, pady = 10, command = prediction_clicked )


my_img = ImageTk.PhotoImage(Image.open("./images/a.jpg"))
maj_logo = Label(root, image = my_img)

ai_image = ImageTk.PhotoImage(Image.open("./images/artifical_inteligence2.png"))
ai_logo = Label(root, image = ai_image)
# 
# ======================================================
#   Grid System
# ======================================================
# 

maj_logo.grid(row = 0, column = 0, columnspan = 3)
title_ASL.grid(row = 1, column = 0, columnspan = 3 )
subtitle.grid(row = 2, column = 0, columnspan = 3 )
subtitle2.grid(row = 3, column = 0, columnspan = 3 )
subtitle3.grid(row = 4, column = 0, columnspan = 3 )
ai_logo.grid(row = 5, column = 0, columnspan = 3 )
register_sign.grid(row = 6, column = 0)
train_model.grid(row = 6, column = 1)
make_prediction.grid(row = 6, column = 2)





root.mainloop()
