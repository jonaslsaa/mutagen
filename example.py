from pprint import pprint
import openai
from pydantic import BaseModel
from mutagen import Mutagen

'''
# Using instructor client:
import instructor
instructor_client = instructor.from_openai(openai.OpenAI(
    base_url="https://api.deepinfra.com/v1/openai",
    api_key="...",
), mode=instructor.Mode.JSON)

mutagen = Mutagen(instructor_client, "meta-llama/Meta-Llama-3-70B-Instruct", use_structured_output=False)
'''

openai_client = openai.OpenAI(
    api_key="...",
)
mutagen = Mutagen(openai_client, "gpt-4o-mini-2024-07-18", use_structured_output=True)
mutagen.config["on_disallowed_mutation"] = "raise" # If the model tries to add/delete from a model, raise an error

# Test mutations of sets
facts = set(["I am a software engineer", "I like the blue color", "I am 25 years old"])
user_message = "I hate the color blue actually, and I like cars!"

print(" * Original facts:")
pprint(facts)
print(" * User message:", user_message)

new_facts, mutations = mutagen.mutate(facts, user_message, "User stated a new fact about themselves, add them! Change or remove if they are incorrect, keep most. There shouln't be contradictions.")
print(" * Mutations:")
pprint(mutations)
pprint(new_facts)
print()

# Test mutations of pydanitc model

class TestModel(BaseModel):
    name: str
    age: int
    color: str
    is_student: bool
    occupation: str | None = None

model = TestModel(name="John", age=25, color="blue", is_student=True)
user_message = "I am not a student, I have become a software engineer!"

print(" * Original model:")
pprint(model.model_dump())
print(" * User message:", user_message)

new_model, mutations = mutagen.mutate(model, user_message, "User stated a new fact about themselves, add them! Only change or remove if they are incorrect.")

print(" * Mutations:")
pprint(mutations)
print(" * New model:")
pprint(new_model.model_dump())