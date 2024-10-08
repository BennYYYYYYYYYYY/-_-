# Matrix(矩陣): 二維張量，有row跟column
'''
1. 矩陣加/減法: 相同位置做運算
'''
import numpy as np 
a = np.array([
    [1, 2, 3],
    [4, 5, 6]
]) # 2維, (2,3)陣列

b = np.array([
    [6, 5, 4],
    [3, 2, 1]
]) # 2維, (2,3)陣列
print(a+b) # 全部為7的(2,3)陣列

'''
2. 矩陣乘法(內積、點積): 
a*b -> a的2維=b的1維 (m,n)*(n,t)=(m,t)
'''
a = np.array([
    [1, 2, 3],
    [4, 5, 6]
]) #(2,3)陣列

b = np.array([
    [9, 8],
    [7, 6],
    [5, 4]
]) #(3,2)陣列
print(a@b) #(2,2)陣列

'''
3. 矩陣中a*b=b*a嗎??  (Ans:不等於)
'''
a = np.array([
    [1, 2],
    [4, 5]
]) #(2,2)陣列

b = np.array([
    [9, 8],
    [7, 6]
]) #(2,2)陣列

print('a*b:', a@b)
print() # 表空一行
print('b*a:',b@a)   # 結果不一樣

'''
4. 轉置矩陣: row與column互換
語法:.T (teanspose)
'''
print(a) 
print(a.T) # 行與列互換

'''
5. 反矩陣
 (1) 必須為方陣(row=column)
 (2) 非奇異方陣(non-singular):所有行和所有列都是線性獨立的。
 [意味著沒有任何一行（或列）可以通過其他行（或列）的線性組合來表示。]
 (3)方陣的行列式不為零時。 

一個方陣是可逆的意味著存在另一個方陣，當這兩個方陣相乘時，結果是單位矩陣。
單位矩陣是一個特殊的方陣，其對角線上的元素全為1，非對角線上的元素全為0。
'''
a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]) #(3,3)陣列

print('逆矩陣:', np.linalg.inv(a)) # 計算a的逆矩陣 (linear algebra inverse)
# Error: a為奇異矩陣(singular), 行列式=0

'''
6. 非奇異矩陣A與其反矩陣的內積=單位矩陣(I)
I1 = 1維單位矩陣
I2 = 2維單位矩陣....
對角線上的元素全為1，非對角線上的元素全為0。
'''
# 驗正: 矩陣@反矩陣=I
a = np.array([
    [9, 8],
    [7, 6]
]) # (2,2)陣列
print(np.round(a @ np.linalg.inv(a))) # 把 a@逆矩陣a 的結果四捨五入到整數
# 結果： 為I2(正確)


'''
7. 驗算結果是否為I3
'''
a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]) 
# a為奇異矩陣:
# 第二行 = 第一行+1
# 第三行 = 第一行+2
# 固不會有反矩陣，進而不會有內積完=單位矩陣的結果
print(np.round(np.linalg.inv(a)))
# Error: 無法算出反矩陣

