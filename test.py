import pickle
import re
import numpy as np
import sklearn
import cv2
import webbrowser
# initalize the cam
cap = cv2.VideoCapture(0)
# initialize the cv2 QRCode detector
detector = cv2.QRCodeDetector()
while True:
    _, img = cap.read()
    # detect and decode
    data, bbox, _ = detector.detectAndDecode(img)
    # check if there is a QRCode in the image
    if data:
        a=data
        break
    # display the result
    cv2.imshow("QRCODEscanner", img)
    if cv2.waitKey(1) == ord("q"):
        break
print(str(a))
url = str(a)
def tokenizer(url):
  """Separates feature words from the raw data
  Keyword arguments:
    url ---- The full URL

  :Returns -- The tokenized words; returned as a list
  """

  # Split by slash (/) and dash (-)
  tokens = re.split('[/-]', url)

  for i in tokens:
    # Include the splits extensions and subdomains
    if i.find(".") >= 0:
      dot_split = i.split('.')

      # Remove .com and www. since they're too common
      if "com" in dot_split:
        dot_split.remove("com")
      if "www" in dot_split:
        dot_split.remove("www")

      tokens += dot_split

  return tokens
model = pickle.load(open('model.pkl', 'rb'))
cVec= pickle.load(open('cVec.pkl', 'rb'))
tVec = pickle.load(open('tVec.pkl', 'rb'))


token_url = tokenizer(url)


c_url = cVec.transform(token_url)
tf_url = tVec.transform(token_url)

unique, counts = np.unique(model.predict(tf_url), return_counts=True)

print("Predicted: ", unique[np.argmax(model.predict(tf_url))])
print("Predicted: ", model.predict(tf_url))