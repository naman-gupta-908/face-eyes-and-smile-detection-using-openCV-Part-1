import cv2

# Load different cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")
# print(face_cascade)

# Read the input image
img = cv2.imread('tests.jpg')
# img = cv2.imread('sample.jpg')


# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(gray)


# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 3)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Detect eyes
    eyes=eye_cascade.detectMultiScale(gray, 1.1, 2)
    
    # Draw rectangle around the eyes
    for (a, b, c, d) in eyes:
        cv2.rectangle(img, (a, b), (a + c, b + d), (255, 0, 0), 2)
    
    # Detect smiles
    smiles=smile_cascade.detectMultiScale(gray, 1.2, 25)
    
    # Draw rectangle around the smiles
    for (m, n, o, p) in smiles:
        cv2.rectangle(img, (m, n), (m + o, n + p), (0, 0, 255), 2)
    
    

# Display the output
cv2.imshow('img', img)
cv2.waitKey()
