from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import cv2
import argparse

def BoxScore(output_data,dwdh,ratio):
    boxes,scores = [],[]
    for i, (x0, y0, x1, y1, oscore) in enumerate(output_data):
        for j in range(len(x0)):
            if oscore[j] >= 0.7:
                
                box = np.array([x0[j]-dwdh[0], y0[j]-dwdh[1], x1[j]-dwdh[0], y1[j]-dwdh[1]])/ratio
                box = box.round().astype(np.int32).tolist()
                
                score = round(float(oscore[j]), 3)
                
                if box not in boxes:
                    boxes.append(box)
                    scores.append(score)
    #return boxes,scores
    for i,k in zip(boxes,scores):
        i.append(k)
    return np.array(i)
    

def ImageBox(image, new_shape=(640, 640), color=(255, 0, 0)):
    
    width, height, channel = image.shape
    
    ratio = min(new_shape[0] / width, new_shape[1] / height)
    
    new_unpad = int(round(height * ratio)), int(round(width * ratio))
    
    dw, dh = (new_shape[0] - new_unpad[0])/2, (new_shape[1] - new_unpad[1])/2

    if (height, width) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return image, ratio, (dw, dh)


model_path = "best_float32.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


video = cv2.VideoCapture(0)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    
    ret, frame = video.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    image,ratio,dwdh = ImageBox(image)
    image = np.expand_dims(image, axis=0)
    image = np.ascontiguousarray(image)
    input_data = image.astype(np.float32)/255
    
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    # boxes,scores = BoxScore(output_data, dwdh, ratio)
    boxes_scores = BoxScore(output_data, dwdh, ratio)
    print(boxes_scores)

    def decrease_box(output_data):
        matrix = output_data[output_data[4]>0.1]
        sorted_matrix_x = matrix[matrix[1].argsort()]
        sorted_matrix_x = sorted_matrix_x[sorted_matrix_x[:, 0].argsort()]
        return sorted_matrix_x

    thresh = 0.3
    dec_box = decrease_box(boxes_scores)

    for (x0,y0,x1,y1,scores) in dec_box[dec_box.shape[0]%10::10]:
        for (x0, y0, x1, y1, scores) in dec_box[dec_box.shape[0]%10::10]:            
            if scores > thresh:
                ymin = int(max(0,x0))
                xmin = int(max(0,y0))
                ymax = int(min(x1,height))
                xmax = int(min(y1,width))

        # ymin = int(max(0,box[0]))
        # xmin = int(max(0,box[1]))
        # ymax = int(min(box[2],height))
        # xmax = int(min(box[3],width))
                cv2.rectangle(frame, (xmin+150,ymin+80), (xmax+120,ymax), (0,255,0), 1)     
    #cv2.imwrite("/content/drive/MyDrive/Colab_Notebooks/Yolo_face_detection/result"+str(i)+".jpg",frame)
    cv2.imshow("input", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
video.release()
cv2.destroyAllWindows()        