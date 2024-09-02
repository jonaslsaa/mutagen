import json
from pprint import pprint
from typing import List, Literal, Set, Tuple, Union, TypeVar, Type
import openai
from pydantic import BaseModel, Field
import instructor

class Mutation(BaseModel):
    type: Literal["add", "set", "remove"]
    key: Union[str, int, None] = Field(description="Path to an element to mutate, for the 'set' and 'remove' mutations")
    new_value: Union[str, int, float, bool, None] = Field(description="The new value of the element, only for the 'add' and 'set' mutations")

class MutationForSetAndList(Mutation):
    type: Literal["add", "set", "remove"]
    key: Union[int, None] = Field(description="Path to an element to mutate, for the 'set' and 'remove' mutations")
    new_value: Union[str, int, float, bool, None] = Field(description="The new value of the element, only for the 'add' and 'set' mutations")

class MutateDict(BaseModel):
    mutations: List[Mutation] = Field(description=f"Mutations to apply to the data structure")

class MutateSetAndList(BaseModel):
    mutations: List[MutationForSetAndList] = Field(description=f"Mutations to apply to the data structure")

InformDataType = Literal["dictonary", "set", "list", "model"]

TModel = TypeVar('T', bound=BaseModel)

class Mutagen:
    def __init__(self, client: instructor.Instructor | openai.OpenAI, llm_model: str, use_structured_output: bool = True):
        self.client = client
        self.is_openai_client = isinstance(client, openai.OpenAI)
        self.use_structured_output = use_structured_output
        if self.use_structured_output and not self.is_openai_client:
            raise ValueError("Structured output is only supported with OpenAI client")
        self.llm_model = llm_model

    def complete_model(self, model: Type[TModel], user_message: str, extra_system_message: str | None = None) -> TModel:
        messages = [
            {"role": "user", "content": user_message},
        ]
        if extra_system_message:
            messages = [
                {"role": "system", "content": extra_system_message},
                *messages,
            ]
        if self.is_openai_client:
            completion = self.client.beta.chat.completions.parse(
                model=self.llm_model,
                messages=messages,
                response_format=model,
            )
            return completion.choices[0].message.parsed
        return self.client.chat.completions.create(
            model=self.llm_model,
            response_model=model,
            messages=messages,
            max_retries=4
        )
    
    def mutate_dict(self, input_dict: dict, user_message: str, extra_system_message: str | None = None, inform_data_type: InformDataType = "dictonary") -> Tuple[dict, List[Mutation]]:
        # Make a copy of the input dictionary so that the model can refer to it
        input_dict_str = json.dumps(input_dict)
        system_message = f"""You are a expert {inform_data_type} mutator, your goal is to mutate users data based on users message.
You will now be given a {inform_data_type} from the user. You will output mutations where you *can* add, set or remove fields. Decide how you should mutate the user's {inform_data_type}.
Only operate on each field once. Never remove then add (set instead).
Mutations must match schema given."""
        if extra_system_message:
            system_message = f"{extra_system_message}\n\n{system_message}"
        mut_cls = MutateDict
        if inform_data_type == "set" or inform_data_type == "list":
            mut_cls = MutateSetAndList
        user_message = f"User's existing {inform_data_type}:\n{input_dict_str}\n\nUser message:\n{user_message}"
        mutations = self.complete_model(mut_cls, user_message, system_message).mutations
        
        # Create a new dictionary with the mutations
        new_dict = input_dict.copy()
        for mutation in mutations:
            if mutation.type == "add":
                next_key = len(new_dict)
                new_dict[next_key] = mutation.new_value
            elif mutation.type == "set":
                new_dict[mutation.key] = mutation.new_value
            elif mutation.type == "remove":
                del new_dict[mutation.key]
        
        return new_dict, mutations

    def mutate_set(self, input_set: set, user_message: str, extra_system_message: str | None = None):
        # Make indexes for the elements in the set so that the model can refer to them
        indexed_set = {index: element for index, element in enumerate(input_set)}
        new_set_indexed, mutations = self.mutate_dict(indexed_set, user_message, extra_system_message, inform_data_type="set")
        return set(new_set_indexed.values()), mutations

    def mutate_list(self, input_list: list, user_message: str, extra_system_message: str | None = None):
        # Make indexes for the elements in the list so that the model can refer to them
        indexed_list = {index: element for index, element in enumerate(input_list)}
        new_list_indexed, mutations = self.mutate_dict(indexed_list, user_message, extra_system_message, inform_data_type="list")
        return list(new_list_indexed.values()), mutations

    def mutate_model(self, input_model: BaseModel, user_message: str, extra_system_message: str | None = None, pass_model_schema: bool = True):
        # Dump the model to a dictionary so that the model can refer to it
        input_model_dict = input_model.model_dump()
        if pass_model_schema:
            extra_system_message = f"{extra_system_message}\n\nUser's model schema:\n{input_model.model_json_schema()}\n"
        # Here we might want to pass the model schema into the extra_system_message, this way the LLM model knows the types of the schema
        new_model_dict, mutations = self.mutate_dict(input_model_dict, user_message, extra_system_message, inform_data_type="model")
        return input_model.__class__.model_validate(new_model_dict), mutations