try:
    from keras.models import load_model
except:
    from keras.saving import load_model
import cv2 , os 
import numpy  as np 



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

model_age =  load_model("./models/age_model.hdf5")
model_gen =  load_model("./models/gender_model.hdf5")

# images_path =  os.path.join("./images")
# images = []


    
    
def predict_age_gender(image):
    gen_array =  ["Male","Female"]
    age_array = ["Child (0-12)","Teenager (12-20)","Adult (20-60)","Senior Citizen (60+) "]
    #img = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    img = np.expand_dims(image.reshape(48,48,1),axis=0)
    age_pred= age_array[np.argmax(model_age.predict(img),axis=1)[0]]
    gen_pred = gen_array[np.argmax(model_gen.predict(img),axis=1)[0]]
    return (age_pred,gen_pred)
    
    
# for i in os.listdir(images_path):
#     img  = cv2.imread(f"./images/{i}")
#     predict_age_gender(img)
#     plt.imshow(img,cmap="gray")
#     plt.show()
    
    
# MALE = 0 FEMALE  = 1
# CHILD = 0 TEENAGER  = 1 ADULT =2 SENIOR_cITIZEN =3 
    
    



