"""Main entrypoint for the app."""
import os
import time
from queue import Queue
from timeit import default_timer as timer

import gradio as gr
from anyio.from_thread import start_blocking_portal

from app_modules.init import app_init
from app_modules.utils import print_llm_response, remove_extra_spaces

llm_loader, qa_chain = app_init()

show_param_settings = os.environ.get("SHOW_PARAM_SETTINGS") == "true"
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

name = "PCI DSS v4"

title = f"""<h1 align="left" style="min-width:200px; margin-top:0;"> Chat with {name} </h1>"""

description_top = f"""\
<div align="left">
<p> Currently Running: <a href="{href}">{model}</a></p>
</div>
"""

description = """\
<div align="center" style="margin:16px 0">
The demo is built on <a href="https://github.com/hwchase17/langchain">LangChain</a>.
</div>
"""

CONCURRENT_COUNT = 1


def qa(chatbot):
    user_msg = chatbot[-1][0]
    q = Queue()
    result = Queue()
    job_done = object()

    def task(question, chat_history):
        start = timer()
        inputs = {"question": question}
        inputs["chat_history"] = chat_history
        ret = qa_chain.call_chain(inputs, None, q)
        end = timer()

        print(f"Completed in {end - start:.3f}s")
        print_llm_response(ret)

        q.put(job_done)
        result.put(ret)

    with start_blocking_portal() as portal:
        chat_history = []
        if chat_history_enabled:
            for i in range(len(chatbot) - 1):
                element = chatbot[i]
                item = (element[0] or "", element[1] or "")
                chat_history.append(item)

        portal.start_task_soon(task, user_msg, chat_history)

        content = ""
        count = 2 if len(chat_history) > 0 else 1

        while count > 0:
            while q.empty():
                print("nothing generated yet - retry in 0.5s")
                time.sleep(0.5)

            for next_token in llm_loader.streamer:
                if next_token is job_done:
                    break
                content += next_token or ""
                chatbot[-1][1] = remove_extra_spaces(content)

                if count == 1:
                    yield chatbot

            count -= 1

        chatbot[-1][1] += "\n\nSources:\n"
        ret = result.get()
        titles = []
        for doc in ret["source_documents"]:
            page = doc.metadata["page"] + 1
            url = f"{doc.metadata['url']}#page={page}"
            file_name = doc.metadata["source"].split("/")[-1]
            title = f"{file_name} Page: {page}"
            if title not in titles:
                titles.append(title)
                chatbot[-1][1] += f"1. [{title}]({url})\n"

        yield chatbot


with open("assets/custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()

with gr.Blocks(css=customCSS) as demo:
    user_question = gr.State("")
    with gr.Row():
        gr.HTML(title)
    gr.Markdown(description_top)
    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            with gr.Row():
                chatbot = gr.Chatbot(elem_id="inflaton_chatbot", height="100%")
            with gr.Row():
                with gr.Column(scale=2):
                    user_input = gr.Textbox(
                        show_label=False,
                        placeholder="Enter your question here",
                        container=False,
                    )
                with gr.Column(
                    min_width=70,
                ):
                    submitBtn = gr.Button("Send")
                with gr.Column(
                    min_width=70,
                ):
                    clearBtn = gr.Button("Clear")
        if show_param_settings:
            with gr.Column():
                with gr.Column(
                    min_width=50,
                ):
                    with gr.Tab(label="Parameter Setting"):
                        gr.Markdown("# Parameters")
                        top_p = gr.Slider(
                            minimum=-0,
                            maximum=1.0,
                            value=0.95,
                            step=0.05,
                            # interactive=True,
                            label="Top-p",
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0,
                            step=0.1,
                            # interactive=True,
                            label="Temperature",
                        )
                        max_new_tokens = gr.Slider(
                            minimum=0,
                            maximum=2048,
                            value=2048,
                            step=8,
                            # interactive=True,
                            label="Max Generation Tokens",
                        )
                        max_context_length_tokens = gr.Slider(
                            minimum=0,
                            maximum=4096,
                            value=4096,
                            step=128,
                            # interactive=True,
                            label="Max Context Tokens",
                        )
    gr.Markdown(description)

    def chat(user_message, history):
        return "", history + [[user_message, None]]

    user_input.submit(
        chat, [user_input, chatbot], [user_input, chatbot], queue=True
    ).then(qa, chatbot, chatbot)

    submitBtn.click(
        chat, [user_input, chatbot], [user_input, chatbot], queue=True, api_name="chat"
    ).then(qa, chatbot, chatbot)

    def reset():
        return "", []

    clearBtn.click(
        reset,
        outputs=[user_input, chatbot],
        show_progress=True,
        api_name="reset",
    )

demo.title = "Chat with PCI DSS v4"
demo.queue().launch(share=share_gradio_app)
