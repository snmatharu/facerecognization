import cv2
import numpy as np
import os

def distance(v1, v2):
	
	return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
	dist = []
	
	for i in range(train.shape[0]):
		
		ix = train[i, :-1]
		iy = train[i, -1]
		
		d = distance(test, ix)
		dist.append([d, iy])
	
	dk = sorted(dist, key=lambda x: x[0])[:k]
	
	labels = np.array(dk)[:, -1]
	
	output = np.unique(labels, return_counts=True)
	index = np.argmax(output[1])
	return output[0][index]


cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

dataset_path ="./FaceData/"
labels = []
class_id = 0
names = {} 
face_data = []
labels = []

for fx in os.listdir(dataset_path):
	if fx.endswith(".npy"):
		names[class_id] = fx[:-4]
		print("Loading file ",fx)
		data_item = np.load(dataset_path+fx)
		face_data.append(data_item)

		target = class_id*np.ones((data_item.shape[0],))
		labels.append(target)
		class_id +=1 


X = np.concatenate(face_data,axis=0)
Y = np.concatenate(labels,axis=0)



print(X.shape)
print(Y.shape)

trainset = np.concatenate((X,Y),axis=1)

#testing
while True:	
	ret,frame=cap.read()
	gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	if ret==False:
		continue

	faces=face_cascade.detectMultiScale(gray_frame,1.3,5)	
	print(faces)
	faces=sorted(faces,key=Lambda f:f[2]*f[3])
	
	for(x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
	

#extract face
		offset=10
		face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section=cv2.resize(face_section,(100,100))
	
	
		out=knn(trainset,face_section.flatten())
		pred_name=name[int(out)]
		CV2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,CV2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h,(0,255,255),2)
	cv2.imshow("Faces",frame)
	key=cv2.waitKey(1)&0xFF
	if	key==ord('q'):
		break

cv2.release()
cv2.destroyAllWindows()
