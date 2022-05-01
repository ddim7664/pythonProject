from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt

#''' p.124
mnist = fetch_mldata('MNIST original',data_home="./")
print(mnist) # mnist 데이터 셋 출력

X, y = mnist["data"], mnist["target"] # X와 y에 mnist의 값을 넣습니다. X가 대문자인 이유는 행렬이기 때문입니다. p.73참조
print(X.shape) # X의 형태 출력 (70000, 784)
print(y.shape) # Y의 형태 출력 (70000, )

print(X)

#''' p.125

some_digit = X[36000] # 70000개중 임의로 하나를 some_digit(784, ) 으로 정의.
some_digit_image = some_digit.reshape(28,28) # (784, 1)을 (28, 28) 로 변경

plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest") #cmap : 그래프 색 옵션 interpolation, interpolation : 픽셀의 경계옵션
plt.axis("off") #기준선 표시 X
plt.show() #화면에 이미지 띄우기