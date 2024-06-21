import cv2
#for face detection use this code
face_cap = cv2.CascadeClassifier("C:/Users/HP/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
video_cap = cv2.VideoCapture(0) #video enable
while True:
    ret, video_data = video_cap.read()  #video capture and image read
    col = cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY)
    faces  = face_cap.detectMultiScale(
          col,
          scaleFactor=1.1,
          minNeighbors=5,
          minSize=(30,30),
          flags =cv2.CASCADE_SCALE_IMAGE
    )
    for(x,y,w,h) in faces:
          cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("video_live",video_data)
    if cv2.waitKey(10) == ord("a"): 
            break
video_cap.release()

'''
Real-Time Face Detection Application

Developed using OpenCV (Python)

Description:

This project creates a real-time face detection application using OpenCV. It leverages a webcam to capture video frames and employs a pre-trained Haar cascade classifier to identify faces within those frames.

Output:

The application displays the live video stream from the webcam on your screen. When a face is detected in the frame, a green rectangle is drawn around it, providing a clear visual confirmation. This allows you to see your own face or anyone else's in the camera's view with a bounding box indicating their presence.

User Experience:

The program runs continuously until you press the "a" key on your keyboard, at which point it stops and releases the webcam resource. This user-friendly control allows you to easily initiate and terminate the face detection process.

Potential Use Cases:

This project serves as a foundation for various computer vision applications that involve face detection, such as:

Security Systems: The application can be integrated into security systems to identify individuals entering or leaving an area. With further development, it could potentially be used for facial recognition purposes.
Human-Computer Interaction: Face detection can be utilized in human-computer interaction systems where a computer program reacts based on the presence or absence of a person's face.
Augmented Reality/Virtual Reality Experiences: By identifying faces, AR/VR applications can tailor their content or experiences to the user's presence.
Video Analysis and Surveillance: This project can be a building block for applications that analyze video footage to detect faces for various purposes, such as drowsiness detection in drivers.
This project demonstrates your ability to build a functional application using OpenCV for real-time face detection. You can further expand on this description by mentioning any specific challenges you encountered and how you overcame them.
'''