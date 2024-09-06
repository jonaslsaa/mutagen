import json
from pprint import pprint
from typing import List, Literal, Set, Tuple, TypedDict, Union, TypeVar, Type
import openai
from pydantic import BaseModel, Field
import instructor

class Mutation(BaseModel):
    type: Literal["add", "set", "delete"]
    key: Union[str, int, None] = Field(description="Path to an element to mutate, for the 'set' and 'delete' mutations")
    new_value: Union[str, int, float, bool, None] = Field(description="The new value of the element, only for the 'add' and 'set' mutations")

class MutationForSetAndList(Mutation):
    type: Literal["add", "set", "delete"]
    key: Union[int, None] = Field(description="Path to an element to mutate, for the 'set' and 'delete' mutations")
    new_value: Union[str, int, float, bool, None] = Field(description="The new value of the element, only for the 'add' and 'set' mutations")

class MutateDict(BaseModel):
    mutations: List[Mutation] = Field(description=f"Mutations to apply to the data structure")

class MutateSetAndList(BaseModel):
    mutations: List[MutationForSetAndList] = Field(description=f"Mutations to apply to the data structure")

InformDataType = Literal["dictonary", "set", "list", "model"]

TModel = TypeVar('T', bound=BaseModel)

MutagenConfigDict = TypedDict('MutagenConfigDict', {
    "on_disallowed_mutation": Literal["raise", "ignore"],
})

class DisallowedMutationError(Exception):
    """Raised when a mutation is not allowed for a given data type."""
    pass

class Mutagen:
    def __init__(self, client: instructor.Instructor | openai.OpenAI, llm_model: str, use_structured_output: bool = True):
        """Initialize Mutagen with a client, model, and output preference."""
        self.client = client
        self.is_openai_client = isinstance(client, openai.OpenAI)
        self.use_structured_output = use_structured_output
        if self.use_structured_output and not self.is_openai_client:
            raise ValueError("Structured output is only supported with OpenAI client")
        self.llm_model = llm_model
        self.config: MutagenConfigDict = {
            "on_disallowed_mutation": "raise",
        }

    def complete_model(self, model: Type[TModel], user_message: str, system_message: str | None = None) -> TModel:
        """Complete a model based on user message and optional system message."""
        messages = [
            {"role": "user", "content": user_message},
        ]
        if system_message:
            messages = [
                {"role": "system", "content": system_message},
                *messages,
            ]
        # Try to use the structured output first
        if self.is_openai_client:
            completion = self.client.beta.chat.completions.parse(
                model=self.llm_model,
                messages=messages,
                response_format=model,
            )
            return completion.choices[0].message.parsed
        # If that fails, try to use instructor
        return self.client.chat.completions.create(
            model=self.llm_model,
            response_model=model,
            messages=messages,
            max_retries=4
        )
        
    def _create_system_message(self, extra_system_message: str | None, _inform_data_type: InformDataType) -> str:
        system_message = f"""You are a expert {_inform_data_type} mutator, your goal is to mutate users data based on users message.
You will now be given a {_inform_data_type} from the user. You will output mutations where you *can* add, set or delete fields. Decide how you should mutate the user's {_inform_data_type}.
Only operate on each field once. Never delete then add (set instead).
Mutations must match schema given."""
        if _inform_data_type == "dictonary" or _inform_data_type == "model":
            system_message = f"{system_message}\nKey must be a string to property in {_inform_data_type}."
        if _inform_data_type == "list" or _inform_data_type == "set":
            system_message = f"{system_message}\nKey must be an index (int) to the element in the list or set. (0-indexed)."
        if _inform_data_type != "model":
            system_message = f" 'add' mutation will add a new element (key property is ignored). {system_message}"
        if _inform_data_type == "model":
            system_message = f" only 'set' mutation is allowed. {system_message}"
        if extra_system_message:
            system_message = f"{extra_system_message}\n\n{system_message}"
        return system_message
    def mutate_dict(self, input_dict: dict, user_message: str, extra_system_message: str | None = None, _inform_data_type: InformDataType = "dictonary") -> Tuple[dict, List[Mutation]]:
        """Mutate a dictionary based on user message and return the new dict and mutations."""
        # Make a copy of the input dictionary so that the model can refer to it
        input_dict_str = json.dumps(input_dict)
        # Create a system message based on the data type
        system_message = self._create_system_message(extra_system_message, _inform_data_type)
        
        # Use the appropriate mutation class based on the data type
        # If the data type is a dict or a model, use MutateDict
        mut_cls = MutateDict
        # If the data type is a set or a list, use MutateSetAndList
        if _inform_data_type == "set" or _inform_data_type == "list":
            mut_cls = MutateSetAndList
        # Create user message with the existing data
        user_message = f"User's existing {_inform_data_type}:\n{input_dict_str}\n\nUser message:\n{user_message}"
        # Get the mutations from the model
        mutations = self.complete_model(mut_cls, user_message, system_message).mutations
        
        # Create a new dictionary with the mutations
        new_dict = input_dict.copy()
        # Apply the mutations to the new dictionary
        for mutation in mutations:
            # Check if the mutation is allowed for the data type
            if _inform_data_type == "model" and mutation.type != "set":
                if self.config["on_disallowed_mutation"] == "raise":
                    raise DisallowedMutationError(f"Mutation type '{mutation.type}' is not allowed for models, only 'set' is allowed.")
                continue
            # Apply the mutation based on the type
            if mutation.type == "add":
                next_key = len(new_dict)
                new_dict[next_key] = mutation.new_value
            elif mutation.type == "set":
                new_dict[mutation.key] = mutation.new_value
            elif mutation.type == "delete":
                del new_dict[mutation.key]
        # Return the new dictionary and the mutations
        return new_dict, mutations

    def mutate_set(self, input_set: set, user_message: str, extra_system_message: str | None = None):
        """Mutate a set based on user message and return the new set and mutations."""
        # Make indexes for the elements in the set so that the model can refer to them
        indexed_set = {index: element for index, element in enumerate(input_set)}
        new_set_indexed, mutations = self.mutate_dict(indexed_set, user_message, extra_system_message, _inform_data_type="set")
        return set(new_set_indexed.values()), mutations

    def mutate_list(self, input_list: list, user_message: str, extra_system_message: str | None = None):
        """Mutate a list based on user message and return the new list and mutations."""
        # Make indexes for the elements in the list so that the model can refer to them
        indexed_list = {index: element for index, element in enumerate(input_list)}
        new_list_indexed, mutations = self.mutate_dict(indexed_list, user_message, extra_system_message, _inform_data_type="list")
        return list(new_list_indexed.values()), mutations

    def mutate_model(self, input_model: BaseModel, user_message: str, extra_system_message: str | None = None, pass_model_schema: bool = True):
        """Mutate a Pydantic model based on user message and return the new model and mutations."""
        # Dump the model to a dictionary so that the model can refer to it
        input_model_dict = input_model.model_dump()
        if pass_model_schema:
            extra_system_message = f"{extra_system_message}\n\nUser's model schema:\n{input_model.model_json_schema()}\n"
        # Here we might want to pass the model schema into the extra_system_message, this way the LLM model knows the types of the schema
        new_model_dict, mutations = self.mutate_dict(input_model_dict, user_message, extra_system_message, _inform_data_type="model")
        return input_model.__class__.model_validate(new_model_dict), mutations
    
    def mutate(self, input_object: Union[dict, set, list, BaseModel], user_message: str, extra_system_message: str | None = None):
        """
        Mutate a Python object based on the user's message.

        This function automatically determines the type of the input object and applies the appropriate mutation strategy.

        Args:
            input_object (Union[dict, set, list, BaseModel]): The object to be mutated.
            user_message (str): The message from the user that guides the mutation.
            extra_system_message (str, optional): Additional context or instructions for the mutation process.

        Returns:
            Tuple[Union[dict, set, list, BaseModel], List[Mutation]]: A tuple containing the mutated object and a list of applied mutations.

        Raises:
            ValueError: If the input_object is of an unsupported data type.

        Note:
            Supported data types are dict, set, list, and Pydantic BaseModel.
        """
        if isinstance(input_object, dict):
            return self.mutate_dict(input_object, user_message, extra_system_message)
        elif isinstance(input_object, set):
            return self.mutate_set(input_object, user_message, extra_system_message)
        elif isinstance(input_object, list):
            return self.mutate_list(input_object, user_message, extra_system_message)
        elif isinstance(input_object, BaseModel):
            return self.mutate_model(input_object, user_message, extra_system_message)
        else:
            raise ValueError(f"Unsupported data type: {type(input_object)}")