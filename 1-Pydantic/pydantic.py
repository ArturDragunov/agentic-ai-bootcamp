#!/usr/bin/env python
# coding: utf-8

# #### Pydantic Basics: Creating and Using Models
# Pydantic models are the foundation of data validation in Python. They use Python type annotations to define the structure and validate data at runtime. Here's a detailed exploration of basic model creation with several examples.
# 
# 

# Data Class vs Normal Class
# 
# Data Class
# 
# - Primarily for storing data
# - Auto-generates __init__, __repr__, __eq__, etc.
# - Less boilerplate code
# - Immutable option available
# 
# Normal Class
# 
# - Full control over behavior
# - Contains methods that perform operations
# - Encapsulation of complex logic
# - Custom initialization
# 
# When to Use Data Class
# 
# - Modeling data structures
# - Need simple value containers
# - Working with DTOs or configuration
# - Primarily storing attributes
# 
# When to Use Normal Class
# 
# - Complex behavior is needed
# - Custom methods dominate
# - Inheritance hierarchies
# - Need fine control over special methods
# 

# In[1]:


from dataclasses import dataclass # data class is a decorator on top of the class to define attributes

@dataclass
class Person(): # in data class we don't need a constructor on the contrary to normal class. It's defined automatically. data class is just for holding values
    name:str
    age:int
    city:str


# In[ ]:


person=Person(name="Krish",age=35,city="Bangalore")
print(person)


# In[ ]:


person=Person(name="Krish",age=35,city=35) # city is int here and we receive no error. Pydantic handles this error
print(person)


# In[4]:


## Pydantic
from pydantic import BaseModel # with pydantic we don't need to use dataclass decorator anymore!


# In[ ]:


class Person1(BaseModel): # class inheriting from BaseModel is called Data Model
    name:str
    age:int
    city:str

person=Person1(name="Krish",age=35,city="Bangalore")
print(person)


# In[ ]:


person1=Person1(name="Krish",age=35,city=35)
print(person1)


# In[ ]:


person2=Person1(name="Krish",age=35,city="35")
print(person2)


# #### 2. Model with Optional Fields
# Add optional fields using Python's Optional type:
# 
# 

# In[8]:


from typing import Optional
class Employee(BaseModel):
    id:int
    name:str
    department:str
    salary: Optional[float]=None #Optional with default value equaling to None
    is_active: Optional[bool]=True #Optional field with default value being True




# In[ ]:


emp1=Employee(id=1,name="John",department="CS")
print(emp1)


# In[ ]:


emp2=Employee(id=2,name="Krish",department="CS",salary="30000") 
# pydantic does type casting where possible!
# It converted "30000" to float 30000.0 without throwing an error
print(emp2)


# Definition:
# - Optional[type]: Indicates the field can be None
# 
# - Default value (= None or = True): Makes the field optional
# 
# - Required fields must still be provided
# 
# - Pydantic validates types even for optional fields when values are provided

# In[ ]:


emp3=Employee(id=2,name="Krish",department="CS",salary="30000",is_active=1) # type casting: 1 transformed to True
print(emp3)


# In[ ]:


emp3=Employee(id=2,name="Krish",department="CS",salary="ban",is_active=100)
print(emp3)


# In[7]:


from typing import List

class Classroom(BaseModel):
    room_number:str
    students: List[str] #List of strings
    capacity:int


# In[ ]:


# Create a classroom
classroom = Classroom(
    room_number="A101",
    students=("Alice", "Bob", "Charlie"), # transformed from tuple to list
    capacity=30
)
print(classroom)


# In[ ]:


list(("Alice", "Bob", "Charlie"))


# In[ ]:


# Create a classroom
classroom1 = Classroom(
    room_number="A101",
    students=("Alice", 123, "Charlie"),
    capacity=30
)
print(classroom1)


# In[ ]:


try:
    invalid_val=Classroom(room_number="A1",students=["Krish",123],capacity=30)

except ValueError as e:
    print(e) # you see even the error location -> students.1


# #### 4. Model with Nested Models
# Create complex structures with nested models:

# In[12]:


from pydantic import BaseModel

class Address(BaseModel):
    street:str
    city:str
    zip_code:str

class Customer(BaseModel):
    customer_id:int
    name:str
    address:Address  ## Nested Model -> address should belong to class Address!


# In[ ]:


customer=Customer(customer_id=1,name="Krish",
                  address={"street":"Main street","city":"Boston","zip_code":"02108"})

print(customer)


# In[ ]:


customer=Customer(customer_id=1,name="Krish",
                  address={"street":"Main street","city":123,"zip_code":"02108"})

print(customer)


# #### Pydantic Fields: Customization and Constraints
# 
# The Field function in Pydantic enhances model fields beyond basic type hints by allowing you to specify validation rules, default values, aliases, and more. Here's a comprehensive tutorial with examples.

# In[ ]:


from pydantic import BaseModel,Field

class Item(BaseModel):
    name:str=Field(min_length=2,max_length=50) # string size
    price:float=Field(gt=0,le=10000)  ## greater than 0 and less than or equal to 10000
    quantity:int=Field(ge=0) # greater or equal to 0

item=Item(name="Book", price=100000,quantity=10)
print(item)


# Pydantic default_factory Summary
# In Pydantic, default_factory=lambda: value provides a function that generates default values dynamically:
# 
# pythonemail: str = Field(default_factory=lambda: "user@example.com", description="Default email address")
# 
# Key points:
# 
# Function is evaluated when an instance is created, not at class definition time
# 
# Primary use-cases:
# 
# Mutable defaults: Prevents shared reference issues
# - BAD: All instances share the same list -> items: list = Field(default=[])
# 
# - GOOD: Each instance gets its own list -> items: list = Field(default_factory=lambda: [])
# 
# Runtime values: For values calculated at instantiation time created_at: datetime = Field(default_factory=lambda: datetime.now())
# 
# Complex defaults: When default requires logic or function calls
# 
# For immutable types like strings, a simple default="value" is usually sufficient unless you need dynamic evaluation.
# 

# In[ ]:


class User(BaseModel):
    username:str=Field(description="Unique username for the user")
    age:int=Field(default=18,description="User age defaults to 18")
    email:str= Field(default_factory=lambda: "user@example.com",description="Default email address")


# Examples
user1 = User(username="alice")
print(user1)




# In[ ]:


user2 = User(username="bob", age=25, email="bob@domain.com")
print(user2)


# In[ ]:


User.model_json_schema()

