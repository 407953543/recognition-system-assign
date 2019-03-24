import numpy as np
import cv2

def myhoughcircle(E,radius,thres):
    accumulator = np.zeros((512,512),np.uint8)

    for i in range(512):
        for j in range(512):
            if E[j][i]:# 255 is edge pixel
                if i-radius < 0:
                    low = 0
                else:
                    low = i-radius
                if i+radius > 512:
                    high = 512
                else:
                    high = i+radius
                for x0 in range(low,high):
                    offset = np.sqrt(np.square(radius)-np.square(i-x0))
                    y1, y2 = int(j+offset), int(j-offset)
                    if y1 >= 0 and y1 <= 511:
                        accumulator[y1][x0] += 1
                    if y2 >= 0 and y2 <= 511:
                        accumulator[y2][x0] += 1

    cv2.imshow('accumulator',accumulator)
    k = cv2.waitKey(0)
    if k == 27:         # 按下esc时，退出
        cv2.destroyAllWindows()

    x, y = [], []
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            if accumulator[j][i] > thres:
                x.append(i)
                y.append(j)
    return y, x

def myfld(input, class1, class2):
    N = np.size(input)
    NC1 = class1.shape[0]
    NC2 = class2.shape[0]
    mean_c1 = np.mean(class1, axis=0, keepdims=True)
    mean_c2 = np.mean(class2, axis=0, keepdims=True)

    S1, S2 = np.zeros((N,N)), np.zeros((N,N))
    for i in range(NC1):
        t = (class1[i] - mean_c1)
        S1 += np.dot(t.T, t)
    for i in range(NC2):
        t = (class2[i] - mean_c2)
        S2 += np.dot(t.T, t)
    s_w = S1 + S2

    w = np.dot(np.linalg.inv(s_w),(mean_c1 - mean_c2).T).T
    w /= np.linalg.norm(w)

    separation_point = np.dot(w, (mean_c1+mean_c2).T) / 2
    if np.dot(w, input.T) > separation_point:
        result = 1
    else:
        result = 2

    return result, w, s_w, mean_c1, mean_c2

#task1
print('*********************************************')
print('Task 1: Hough Circle Detection')
print('*********************************************')
I = cv2.imread('qiqiu.png',0)
E = cv2.Canny(I, 50, 150)

radius = 114
thres = 125
y, x = myhoughcircle(E,radius,thres)

canvas = np.zeros((512,512,3),np.uint8)
for _ in range(len(x)):
    canvas = cv2.circle(canvas, (x[_],y[_]), radius, (255,255,255), 1)
cv2.imshow('image03',canvas)
k = cv2.waitKey(0)
if k == 27:         # 按下esc时，退出
    cv2.destroyAllWindows()

#task2
print('*********************************************')
print('Task 2: Fisher Linear Discriminant')
print('*********************************************')
class1_samples=np.array([[1,2],[2,3],[3,3],[4,5],[5,5]])
class2_samples=np.array([[1,0],[2,1],[3,1],[3,2],[5,3],[6,5]])
input_sample=np.array([2,5]).reshape(1,2)
output_class, w, s_w, mean_c1, mean_c2 = myfld(input_sample, class1_samples, class2_samples)
print('Output class is '+str(output_class)+'.')
print('Within-class scatter matrix:')
print(s_w)
print('Weights:')
print(w)

