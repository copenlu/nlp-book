#!/usr/bin/env python
# coding: utf-8

# ## Comments

# In[1]:


# This is a comment

"""This is a 
    multiline
    comment."""


# ## Variables and Types
# 
# 

# In[2]:


coolvar = 7 # declaration
coolvar # almost the same as print(myvar)


# In[3]:


print(7, type(7)) # integer
print(float(7), type(7.0)) # float


# In[4]:


# ALL these are strings

print("Let's go", type("Let's go")) # you can also escape \' when in single quotes

print('"To be or not to be?"', type("'To be or not to be?'"))

a = """This is a multiline 
string"""
print(a, type(a))


# ## Operators

# In[5]:


a = 73
b = 42

# addition
print('a + b =', a + b)

# subtraction
print('a - b =', a - b)

# multiplication
print('a * b =', a * b)

# classic division - returns a float
print('a / b =', a / b)

# floor division - discards the fractional part
print('a // b =', a // b)

# remainder of the division
print('a % b =', a % b)

# power (a^2)
print('a ** 2 =', a ** 2)


# In[6]:


print(2 < 5)
print(4 > 10)
print(3 >= 3)
print(5 == 6)
print(6 != 9)


# ## Implicit Type Conversion - only for some types

# In[7]:


num_int = 123  # integer type
num_flo = 1.23 # float type
num_int + num_flo


# In[8]:


num_int = 123     # int type
num_str = "456"   # str type

num_int+num_str


# In[ ]:


str(num_int) + num_str


# ## Data Structures

# In[ ]:


my_list = [] # empty list
my_list = [1, 2, 3] # list of integers
my_list = [1, "Hello", 3.4] # list with mixed data types


# In[ ]:


print(my_list[0]) # Accessing first element


# In[ ]:


print(my_list[:])
print(my_list[0:2])
print(my_list[-3:-1])
print(my_list[1:])
print(my_list[::2])


# In[ ]:


language = ("French", "German", "English", "Polish")


# In[ ]:


str = 'this is a string'


# In[ ]:


# set of integers
my_set = set()
my_set = {1, 2, 3}
print(my_set)

# set of mixed datatypes
my_set = {1.0, "Hello", (1, 2, 3)}
print(my_set)


# In[ ]:


# set of integers
my_set = {1, 2, 3}

my_set.add(4)
print(my_set) # Output: {1, 2, 3, 4}

my_set.add(2)
print(my_set) # Output: {1, 2, 3, 4}

my_set.update([3, 4, 5])
print(my_set) # Output: {1, 2, 3, 4, 5}

my_set.remove(4)
print(my_set) # Output: {1, 2, 3, 5}


# In[ ]:


# empty dictionary
my_dict = {}

# dictionary with integer keys
my_dict = {1: 'apple', 2: 'ball'}

# dictionary with mixed keys
my_dict = {'name': 'John', 1: [2, 4, 3]}

person = {'name':'Jack', 'age': 26, 'salary': 4534.2}
print(person['age'])


# ## Python Control Flow

# In[ ]:


num = -1

if num > 0:
    print("Positive number")
elif num == 0:
    print("Zero")
else:
    print("Negative number")


# In[ ]:


if False:
    print("I am inside the body of if.")
    print("I am also inside the body of if.")
print("I am outside the body of if")


# In[ ]:


n = 100

# initialize sum and counter
sum = 0
i = 1

while i <= n:
    sum = sum + i
    i = i+1    # update counter

print("The sum is", sum)


# In[ ]:


numbers = [6, 5, 3, 8, 4, 2]

sum = 0

# iterate over the list
for val in numbers:
    sum = sum+val

print("The sum is", sum) # Output: The sum is 28


# ## Functions

# In[ ]:


def print_lines():
    print("I am line1.")
    print("I am line2.")

# function call
print_lines()


# In[ ]:


def add_numbers(a, b):
    sum = a + b
    return sum

result = add_numbers(4, 5)
print(result)


# In[ ]:


def greet(name, msg = "Good morning!"):
    """
    This function greets to
    the person with the
    provided message.

    If message is not provided,
    it defaults to "Good
    morning!"
    """

    print("Hello",name + ', ' + msg)

greet("Kate")
greet("Bruce","How do you do?")


# In[ ]:


def greet(*names):
    """This function greets all
    the person in the names tuple."""

    # names is a tuple with arguments
    for name in names:
        print("Hello",name)

greet("Monica","Luke","Steve","John")


# ## Python OOP

# In[ ]:


class ComplexNumber:
    def __init__(self,r = 0,i = 0):  # constructor
        self.real = r
        self.imag = i

    def getData(self):
        print("{0}+{1}j".format(self.real,self.imag))


c1 = ComplexNumber(2,3) # Create a new ComplexNumber object
c1.getData()

c2 = ComplexNumber() # Create a new ComplexNumber object
c2.getData()


# ## List Comprehension vs For Loop in Python
# 
# [expression for item in list]

# In[ ]:


h_letters = []

for letter in 'human':
    h_letters.append(letter)

print(h_letters)

h_letters = [letter for letter in 'human']
print( h_letters)


# In[ ]:


number_list = [ x for x in range(20) if x % 2 == 0]
print(number_list)


# In[ ]:


matrix = [[1, 2], [3,4], [5,6], [7,8]]
transpose = [[row[i] for row in matrix] for i in range(2)]
print (transpose)


# ## Miscelanneous

# In[ ]:


list(range(1, 10))

