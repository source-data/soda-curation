import openai
from typing import List, Dict
from .general import StructureZipFile, ZipStructure
from openai.types.beta.thread import Thread
from .prompts import get_structure_zip_prompt

class StructureZipFileGPT(StructureZipFile):
    def __init__(self, config: Dict):
        self.config = config
        self._client = openai.Client(api_key=self.config['api_key'])
        self._assistant = self._client.beta.assistants.retrieve(
            config["structure_zip_assistant_id"]
        )
        
        # Update assistant instructions
        prompt = get_structure_zip_prompt(
            file_list="[File list will be provided in each request]",
            custom_instructions=self.config.get('custom_prompt_instructions')
        )
        self._assistant = self._client.beta.assistants.update(
            config["structure_zip_assistant_id"],
            model=self.config['model'],
            temperature=self.config['temperature'],
            top_p=self.config['top_p'],
            instructions=prompt
        )

    def _prepare_query(self, file_list: List[str]) -> Thread:
        thread = self._client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Process this file list:\n{file_list}",
                }
            ]
        )
        return thread

    def _run_prompt(self, file_list: List[str]) -> str:
        """
        Process the input user prompt using the AI assistant.

        Parameters:
            file_list (List[str]): List of files in the Zip file.

        Returns:
            str: JSON string from the AI assistant.
        """
        thread = self._prepare_query(f"""[{", ".join(file_list)}]""")

        run = self._client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=self._assistant.id,
        )

        if run.status == 'completed':
            messages = self._client.beta.threads.messages.list(
                thread_id=thread.id
            )
            result = messages.data[0].content[0].text.value
            return result
        else:
            return run.status

    def process_zip_structure(self, file_list: List[str]) -> ZipStructure:
        try:
            thread = self._prepare_query(f"""[{", ".join(file_list)}]""")

            run = self._client.beta.threads.runs.create_and_poll(
                thread_id=thread.id,
                assistant_id=self._assistant.id,
            )

            if run.status == 'completed':
                messages = self._client.beta.threads.messages.list(
                    thread_id=thread.id
                )
                result = messages.data[0].content[0].text.value
                print(f"Debug - AI response: {result}")  # Debug print
                return self._json_to_zip_structure(result)
            else:
                print(f"Debug - AI response: {run.status}")  # Debug print
                return run.status

        except Exception as e:
            print(f"Error in AI processing: {str(e)}")
            return None

