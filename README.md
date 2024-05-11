# Mutagen

MutaGen is a Python library that makes it easy to dynamically mutate data structures (dictionaries, sets, lists, and models) based on user input. Built on top of Pydantic and leveraging the Instructor library, which simplifies working with structured outputs from large language models, MutaGen provides a flexible and efficient way to update data structures in response to user input, ensuring schema-adherent mutations that maintain data integrity. Get ready to supercharge your data manipulation workflows!

## Key Features

* AI-Powered Mutations: Leverages OpenAI-compatible language models to generate intelligent data structure mutations based on user input.
* Schema-Adherent Updates: Ensures that mutated data structures adhere to their original schema, maintaining data integrity and consistency.
* Flexible Data Support: Supports mutations for dictionaries, sets, lists, and custom models defined using Pydantic.
Efficient and Scalable: Built on top of Pydantic and leveraging the Instructor library, MutaGen provides a lightweight and efficient solution for dynamic data manipulation.
* Simple API: Offers a simple and intuitive API for integrating MutaGen into your application, making it easy to get started with AI-powered data mutations.

## Example

### Mutate a dictionary

```python
# Create a Mutagen instance with an Instructor client and an LLM model
mutagen = Mutagen(instructor_client, "meta-llama/Meta-Llama-3-70B-Instruct")

# Define a dictionary to mutate
input_dict = {"name": "John", "age": 25, "color": "blue"}

# Define a user message to guide the mutation
user_message = "I hate the color blue, and I'm 30 years old now!"

# Mutate the dictionary using the user message
new_dict, mutations = mutagen.mutate_dict(input_dict, user_message)

# new_dict: {"name": "John", "age": 30}
# mutations: [
#     Mutation(type='set', key=['age'], value=30),
#     Mutation(type='remove', key=['color'])
# ]
```

### Mutate a Pydantic model

```python
from pydantic import BaseModel

# Define a Pydantic model
class Person(BaseModel):
    name: str
    occupation: str

# Create an instance of the model
person = Person(name="John", occupation="Engineer")

# Define a user message to guide the mutation
user_message = "I want to change my occupation to Data Scientist."

# Mutate the model using the user message
new_person, mutations = mutagen.mutate_model(person, user_message)

# new_person: Person(name='John', occupation='Data Scientist')
# mutations: [Mutation(type='set', key=['occupation'], value='Data Scientist')]
```
