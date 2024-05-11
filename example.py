from pprint import pprint
import instructor
import openai
from pydantic import BaseModel
from mutagen import Mutagen

instructor_client = instructor.from_openai(openai.OpenAI(
    base_url="https://api.deepinfra.com/v1/openai",
    api_key="...",
), mode=instructor.Mode.JSON)

mutagen = Mutagen(instructor_client, "meta-llama/Meta-Llama-3-70B-Instruct")

# Test mutations of sets
facts = set(["I am a software engineer", "I like the blue color", "I am 25 years old"])
user_message = "I hate the color blue actually, and I like cars!"

print(" * Original facts:")
pprint(facts)
print(" * User message:", user_message)

new_facts, mutations = mutagen.mutate_set(facts, user_message, "User stated a new fact about themselves, add them! Change or remove if they are incorrect, keep most. There shouln't be contradictions.")
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

model = TestModel(name="John", age=25, color="blue", is_student=True)
user_message = "I am not a student, I am a teacher now!"

print(" * Original model:")
pprint(model.model_dump())
print(" * User message:", user_message)

new_model, mutations = mutagen.mutate_model(model, user_message, "User stated a new fact about themselves, add them! Only change or remove if they are incorrect.")

print(" * Mutations:")
pprint(mutations)
print(" * New model:")
pprint(new_model.model_dump())