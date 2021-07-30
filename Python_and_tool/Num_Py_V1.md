# NUMPY

### 1. INTRODUCE NUMPY
Numpy là thư viện nổi tiếng để xử lý dữ liệu trong python, nó hỗ trợ cực kì mạnh mẽ cho Machine-learning, Deep-learning, dễ dàng chuyển đổi giữa các fame-work nổi tiếng như Tensoflow, Pytorch...


![image](https://user-images.githubusercontent.com/42260182/99546541-174d6900-29e9-11eb-83a4-6ddd247f013c.png)

Ví dụ: List 1D
```python
import numpy as np
list1D = [1,2,3]
data = np.array(list1D)
print(data)
print(data.shape)
```
kết quả:
```
[1 2 3]
(3,)
```

Ví dụ: List 2D
```python
import numpy as np
list2D = [[1,2],[3,4],[5,6]]
data = np.array(list2D)
print(data)
print(data.shape)
```
kết quả:
```
[[1 2] [3 4] [5 6]]
(3, 2)
```

Ví dụ: List 3D
```python
import numpy as np
list3D = [[[1,6],[2,2],[3,4]],
[[4,7],[5,2],[6,9]],
[[7,7],[8,2],[9,5]]]
data = np.array(list3D)
data.shape
```

**2. CREATE NUMPY ARRAY**

* 2.1 Hàm zeros()
Hàm zeros() tạo ra một numpy array với toàn bộ các phần tử là 0
Ví dụ:
```python
import numpy as np
#Tạo một numpy array 2 dòng 3 cột với toàn bộ các phần tử bằng 0
arr = np.zeros((2,3))
print(arr)
```
Kết quả:
```
[[0. 0. 0.] 
 [0. 0. 0.]]
```
* 2.2 Hàm ones()
Hàm ones() tạo ra một numpy array với toàn bộ các phần tử là 1
Ví dụ:
```python
import numpy as np
#Tạo một numpy array 2 dòng 3 cột với toàn bộ các phần tử bằng 1
arr = np.ones((2,3))
print(arr)
```
Kết quả:
```
[[1. 1. 1.] 
 [1. 1. 1.]]
```
* 2.3 Hàm full()
Hàm full() tạo ra một numpy array với toàn bộ các phần tử là là hằng số nó tương tự như hàm memset() trong c++ 
```python
import numpy as np 
arr = np.full((2,3),10)
print(arr)
```
Kết quả:
```
[[10 10 10] 
 [10 10 10]]
```
* 2.4 Hàm arange()

Ví dụ:
```python
import numpy as np
#np.arrange(start =, stop = , step =)
arr1 = np.arange(5)
print(arr1)
#np.arrange(start = 0, stop = 7, step = 2)
arr2 = np.arrange(0,7,2)
print(arr2)
```
Kết quả:
```
[0 1 2 3 4] 
[0 2 4 6]
```
* 2.5 Hàm eye()
Tạo ra một numpy array với đường chéo là số 1, các số còn lại đều bằng 0

Ví dụ
```python
import numpy as np
arr = np.eys(3)
print(arr)
```

Kết quả:
```
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```
* 2.6 Hàm random() 

Ví dụ:
```python
import numpy as np
arr = np.random.random((2,3))
print(arr)
```
**3 Một số hàm quan trọng trong numpy**
* 3.1 Hàm where()

![image](https://user-images.githubusercontent.com/42260182/99984823-2153db00-2de0-11eb-904f-60ca7c64ae8e.png)

* 3.1 Flatten() 
Flaten chuyển từ array nhiều chiều sang array 1 chiều
![image](https://user-images.githubusercontent.com/42260182/99984983-4fd1b600-2de0-11eb-98ae-2423c3ee28d8.png)

* 3.2 Reshape()

Reshape là một hàm rất quan trọng trong quá trình xử lý dữ liệu chuẩn bị cho quá trình training mục đích của nó là chuyển đổi các chiều trong dữ liệu

ví dụ:
```python
import numpy as np

l = [[1,2,3],
[4,5,6]]
data = np.array(l)
print(data.shape)
data_reshape = np.reshape(data,(3,2))
print(data_reshape)
data_reshape_1 = np.reshape(data,(1,6))
print(data_reshape_1)
```

Trong hàm reshape() muốn chia thành kiểu dữ liệu khác kiểu bản đầu thì điều kiện là a*b = c*d với (a,b) là chiều ban đầu và (c,d) là chiều muốn chuyển đổi

**4. Array Indexing**
* 4.1 Slicing
![image](https://user-images.githubusercontent.com/42260182/100472071-21b8e280-310e-11eb-9dd2-33cf14273ff4.png)


Lưu ý: axis là số chiều, dấu "," dùng để ngăn cách từng chiều, dấu ":" dùng để xác định số phần tử trong chiều đó

Ví dụ:
```python
import numpy as np
a_arr = np.array([[1,2,3],
[5,6,7]])
print(a_arr)
#b_array lấy các dòng cột 1 và cột 2
b_arr = a_arr[:,1:3]
print(b_arr)
``` 

Kết quả:
```
[[1 2 3] 
 [5 6 7]]
[[2 3] 
 [6 7]]
```
* 4.1 Lấy một hàng trong numpy

ví dụ:
```python
import numpy as np

arr = np.array([[1,2,3],
                [4,5,6],
                [7,8,9]])
#cách truy cập 1
row_1 = arr[1,:]
print(row_1,row_1.shape)
#cách truy cập 2
row_2 = arr[1:2,:]
print(row_2,row_2.shape)
```

Kết quả:
```
[4 5 6] (3,) 
[[4 5 6]] (1, 3)
```
* 4.2 Lấy một cột trong numpy

ví dụ:
```python
import numpy as np
arr = np.array([[1,2,3],
                [4,5,6],
                [7,8,9]])
#cách truy cập 1
colum_1 = arr[:,1]
print(colum_1,colum_1.shape)
#cách truy cập 2
colum_2 = arr[:,1:2]
print(colum_2,colum_2.shape)
```
Kết quả:
```
[2 5 8] (3,) 
[[2] [5] [8]] (3, 1)
```

* 4.3 Lấy với điều kiện

Ví dụ: Tìm các phần tử trong numpy > 5

```python
import numpy as np
arr = np.arange(9)
mask = arr>5
output = arr[mask]
output
```
Kết quả:
```
array([6, 7, 8])
```
**5. CÁC PHÉP TOÁN TRONG NUMPY**

* 5.1 Adition
* ![image](https://user-images.githubusercontent.com/42260182/100513889-f8966180-31a2-11eb-94b6-6eb42d9908a8.png)

Ví dụ:
```python
x = np.array([1,2,3,4])
y = np.array([4,5,6,7])
print(x)
print(y)
#Tổng 2 mảng
print(x+y)
#Sử dụng add funcition
print(np.add(x,y))
```
Kết quả:
```
[1 2 3 4] 
[4 5 6 7] 
[ 5 7 9 11] 
[ 5 7 9 11]
```
* 5.2 Phép nhân

![image](https://user-images.githubusercontent.com/42260182/100513950-88d4a680-31a3-11eb-8a80-dd597ed542e7.png)

 -Phép nhân trong numpy là phép nhân từng phần tử với phần tử trong numpy
Ví dụ:
```python
import numpy as np
x = np.array([1,2,3,4,5])
y = np.array([3,4,5,6,7])
print(x)
print(y)
#phép nhân
print(x*y)
#hàm
print(np.multiply(x,y))
```

Kết quả:
```
[1 2 3 4 5]
[3 4 5 6 7] 
[ 3 8 15 24 35] 
[ 3 8 15 24 35]
```

* 5.3 Phép chia
-Phép chia trong numpy là phép chia từng phần tử với phần tử trong numpy
Ví dụ:
```python
import numpy as np
x = np.array([1,2,3,4,5])
y = np.array([5,6,7,8,9])
#phép chia
print(x/y)
#chia lấy phần nguyên
print(x//y)
#dùng hàm
print(np.divide(x,y))
```
Kết quả:
```
  

import numpy as np

x = np.array([1,2,3,4,5])

y = np.array([5,6,7,8,9])

#phép chia

print(x/y)

#chia lấy phần nguyên

print(x//y)

#dùng hàm

print(np.divide(x,y))
[0.2        0.33333333 0.42857143 0.5        0.55555556]
[0 0 0 0 0]
[0.2        0.33333333 0.42857143 0.5        0.55555556]
```

* 5.4 Căn bậc 2

```python
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
print(np.sqrt(x))
```

Kết quả:
```
[1.         1.41421356 1.73205081 2.         2.23606798 2.44948974
 2.64575131 2.82842712 3.         3.16227766]
```

* 5.5 Inner product

Ví dụ:
```python
import numpy as np
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])
print(x.dot(y))
print(np.dot(x,y))
```
Kết quả:
```
55
55
```
* 5.6 Vector-matrix multiplication
![image](https://user-images.githubusercontent.com/42260182/100514407-08b04000-31a7-11eb-93d6-e8379a681aec.png)

Ví dụ:
```python
import numpy as np
x = np.array([[1,2],
              [3,4])
y = np.array([1,2])

print(x)
print(y)
print(x.dot(y))
print(y.dot(x))
```

Kết quả:
```
[[1 2]
 [3 4]] 
 [1 2] 
[ 5 11] 
[ 7 10]
```

* 5.7 Matrix-matrix multiplication
![image](https://user-images.githubusercontent.com/42260182/100514553-377ae600-31a8-11eb-98bc-58c23786efaa.png)

Ví dụ:
```python
x = np.array([[1,2],
              [3,4]])
y = np.array([[1,1],
              [1,1]])
print(x,x.shape)
print(y,y.shape)
print(x.dot(y))
print(y.dot(x))
```

Kết quả:
```
[[1 2] 
 [3 4]] (2, 2) 
[[1 1] 
 [1 1]] (2, 2)
[[3 3]
 [7 7]] 
[[4 6] 
 [4 6]]
```
* 5.8 Transpose (Chuyển vị)
![image](https://user-images.githubusercontent.com/42260182/100514607-c8ea5800-31a8-11eb-9814-e0c3da169045.png)


* 5.9 SUM
![image](https://user-images.githubusercontent.com/42260182/100514623-f6370600-31a8-11eb-8f1d-1f09cbb3f1e3.png)

* 5.10 Max and Min
![image](https://user-images.githubusercontent.com/42260182/100514662-431adc80-31a9-11eb-947a-d34dfae3dda4.png)


**6. Broadcasting**

Broadcasting là các phép toán numpy thực hiện khi số chiều không tương thích
![image](https://user-images.githubusercontent.com/42260182/100514834-701bbf00-31aa-11eb-8765-b4be44bafaa4.png)

Factor = 2 tương ứng sẽ tạo ra dạng [2,2,2]

Tương tự như ma trận 2 chiều hoặc nhiều chiều hơn nữa
![image](https://user-images.githubusercontent.com/42260182/100514857-a6f1d500-31aa-11eb-862e-b66239fb22da.png)

**7. Data Processing**
Iris dataset
![image](https://user-images.githubusercontent.com/42260182/100515476-1b2e7780-31af-11eb-8d24-a66c2f7ede4e.png)

+Code xử lý 
```python
import numpy as np

import numpy.core.defchararray as np_f

  

X = np.genfromtxt('IRIS.csv',delimiter=',',dtype = 'float', usecols = [0,1,2,3],skip_header=1)
print(X.shape)
Y = np.genfromtxt('IRIS.csv',delimiter=',',dtype ='str', usecols = 4,skip_header=1) 
categories = np.unique(y)
for i in  range(categories.size):
y = np_f.replace(y,categories[i],str(i)) 
y=y.astype('float')
print(y)
```

Kết quả:
![image](https://user-images.githubusercontent.com/42260182/100515687-7e6cd980-31b0-11eb-9442-c9d94ffaa78a.png)

**TỔNG KẾT**

![image](https://user-images.githubusercontent.com/42260182/100515717-b5db8600-31b0-11eb-98bb-830c4cad3398.png)
