import os
import sys
from timeit import default_timer as timer

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

from app_modules.init import app_init
from app_modules.utils import print_llm_response

llm_loader, qa_chain = app_init()


class MyCustomHandler(BaseCallbackHandler):
    def __init__(self):
        self.reset()

    def reset(self):
        self.texts = []

    def get_standalone_question(self) -> str:
        return self.texts[0].strip() if len(self.texts) > 0 else None

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Run when chain ends running."""
        print("\n<on_llm_end>")
        # print(response)
        self.texts.append(response.generations[0][0].text)


chatting = len(sys.argv) > 1 and sys.argv[1] == "chat"
questions_file_path = os.environ.get("QUESTIONS_FILE_PATH")
chat_history_enabled = os.environ.get("CHAT_HISTORY_ENABLED") or "true"

custom_handler = MyCustomHandler()

# Chatbot loop
chat_history = []

# Open the file for reading
file = open(questions_file_path, "r")

# Read the contents of the file into a list of strings
questions = file.readlines()
for i in range(len(questions)):
    questions[i] = questions[i].strip()

# Close the file
file.close()

if __name__ == "__main__":
    questions.append("exit")

    chat_start = timer()

    while True:
        if chatting:
            query = input("Please enter your question: ")
        else:
            query = questions.pop(0)

        query = query.strip()
        if query.lower() == "exit":
            break

        print("\nQuestion: " + query)
        custom_handler.reset()

        start = timer()
        result = qa_chain.call_chain(
            {"question": query, "chat_history": chat_history},
            custom_handler,
            None,
            True,
        )
        end = timer()
        print(f"Completed in {end - start:.3f}s")

        if chat_history_enabled == "true":
            chat_history.append((query, result["answer"]))

    chat_end = timer()
    total_time = chat_end - chat_start
    print(f"Total time used: {total_time:.3f} s")
    print(f"Number of tokens generated: {llm_loader.streamer.total_tokens}")
    print(
        f"Average generation speed: {llm_loader.streamer.total_tokens / total_time:.3f} tokens/s"
    )
