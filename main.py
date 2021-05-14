import cv2
thres = 0.5

 #img = cv2.imread('unnamed.jpg')

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

classNames = []
classFile = 'labels.txt'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

#ref - https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
configFile = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' #loading the config
frozenModel = 'frozen_inference_graph.pb'

#Loading model
model = cv2.dnn_DetectionModel(frozenModel,configFile)
#setting up the configuration
model.setInputSize(320,320)
model.setInputScale(1.0/127.5) ##255/2 = 127.5
model.setInputMean((127.5,127.5,127.5)) ##
model.setInputSwapRB(True)

while True:

        success,img = cap.read()

        ClassIndex,confidece, bbox = model.detect(img,confThreshold=thres)
        print(ClassIndex,bbox)

        # to avoid empty frames
        if len(classNames)!=0:

            for ClassInd, conf,box in zip(ClassIndex.flatten(), confidece.flatten(), bbox):

                cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                cv2.putText(img,classNames[ClassInd-1],(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.putText(img, str(round(conf*100)), (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 255, 0), 2)

        cv2.imshow("output",img)
        cv2.waitKey(1)
