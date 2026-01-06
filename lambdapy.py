# lambda x:x + 10
 
square = lambda x: x * x
 
 
r = square(5)
print(r)
 
lambda_add = lambda a, b: a + b
 
result = lambda_add(3, 7)
 
print(result)
 
numbers = [1, 2, 3, 4, 5]
 
squared = list(map(lambda x: x ** 2, numbers))
sq=(lambda x: x ** 2, numbers)
print(sq)
sqmap= map(lambda x: x ** 2, numbers)
print(sqmap) 
print(squared)
 
 
evens = list(filter(lambda x: x % 2 == 0, numbers))
evenfilter = filter(lambda x: x % 2 == 0, numbers)
print(evenfilter)
print(evens)
 