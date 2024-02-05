"""Main entrypoint for the app."""

import os
from threading import Thread
import time
from queue import Queue
from timeit import default_timer as timer

import gradio as gr

os.environ["USER_CONVERSATION_SUMMARY_BUFFER_MEMORY"] = "true"

from app_modules.init import app_init
from app_modules.utils import print_llm_response

llm_loader, qa_chain = app_init()

share_gradio_app = os.environ.get("SHARE_GRADIO_APP") == "true"
using_openai = os.environ.get("LLM_MODEL_TYPE") == "openai"
chat_history_enabled = os.environ.get("CHAT_HISTORY_ENABLED") == "true"

model = (
    "OpenAI GPT-3.5"
    if using_openai
    else os.environ.get("HUGGINGFACE_MODEL_NAME_OR_PATH")
)
href = (
    "https://platform.openai.com/docs/models/gpt-3-5"
    if using_openai
    else f"https://huggingface.co/{model}"
)

title = "Chat with PCI DSS v4"

questions_file_path = os.environ.get("QUESTIONS_FILE_PATH")

# Open the file for reading
with open(questions_file_path, "r") as file:
    examples = file.readlines()
    examples = [example.strip() for example in examples]

description = f"""\
<div align="left">
<p> Currently Running: <a href="{href}">{model}</a></p>
</div>
"""


def task(question, chat_history, q, result):
    start = timer()
    inputs = {"question": question, "chat_history": chat_history}
    ret = qa_chain.call_chain(inputs, None, q)
    end = timer()

    print(f"Completed in {end - start:.3f}s")
    print_llm_response(ret)

    result.put(ret)


def predict(message, history):
    print("predict:", message, history)

    chat_history = []
    if chat_history_enabled:
        for element in history:
            item = (element[0] or "", element[1] or "")
            chat_history.append(item)

    q = Queue()
    result = Queue()
    t = Thread(target=task, args=(message, chat_history, q, result))
    t.start()  # Starting the generation in a separate thread.

    partial_message = ""
    count = 2 if len(chat_history) > 0 else 1

    while count > 0:
        while q.empty():
            print("nothing generated yet - retry in 0.5s")
            time.sleep(0.5)

        for next_token in llm_loader.streamer:
            partial_message += next_token or ""
            # partial_message = remove_extra_spaces(partial_message)
            yield partial_message

        if count == 2:
            partial_message += "\n\n"

        count -= 1

    partial_message += "\n\nSources:\n"
    ret = result.get()
    titles = []
    for doc in ret["source_documents"]:
        page = doc.metadata["page"] + 1
        url = f"{doc.metadata['url']}#page={page}"
        file_name = doc.metadata["source"].split("/")[-1]
        title = f"{file_name} Page: {page}"
        if title not in titles:
            titles.append(title)
            partial_message += f"1. [{title}]({url})\n"

    yield partial_message


# Setting up the Gradio chat interface.
gr.ChatInterface(
    predict,
    title=title,
    description=description,
    examples=examples,
).launch(
    share=share_gradio_app
)  # Launching the web interface.
