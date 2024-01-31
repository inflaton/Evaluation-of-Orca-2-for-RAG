import ast
import codecs
import json
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from datasets import Dataset
from langchain_openai.chat_models import ChatOpenAI


from app_modules.init import app_init

llm_loader, qa_chain = app_init()
qa = qa_chain.get_chain()

gpt4_turbo = ChatOpenAI(model_name="gpt-4-turbo-preview")


def load_notebook(filename, print_source=False):
    f = codecs.open(filename, "r")
    source = f.read()

    print("loading: ", filename)
    notebook = json.loads(source)

    if print_source:
        pySource = f"### Python code from {filename}:\n"
        for x in notebook["cells"]:
            for x2 in x["source"]:
                pySource = pySource + x2
                if x2[-1] != "\n":
                    pySource = pySource + "\n"

        print(pySource)
    return notebook


def parse_outputs(outputs):
    questions = [
        "Question: What's PCI DSS?\n",
        "Question: Can you summarize the changes made from PCI DSS version 3.2.1 to version 4.0?\n",
        "Question: new requirements for vulnerability assessments\n",
        "Question: more on penetration testing\n",
    ]
    result = []
    for question in questions:
        start = outputs.index(question)
        conversation = ast.literal_eval(outputs[start + 1])
        index = start + 2

        if len(conversation["chat_history"]) > 0:
            conversation["standalone_question"] = ""
            while not outputs[index].startswith("<on_llm_end>"):
                conversation["standalone_question"] += outputs[index]
                index += 1

            index += 1
            while outputs[index] == "\n":
                index += 1

        conversation["answer"] = ""

        while not outputs[index].startswith("<on_llm_end>"):
            conversation["answer"] += outputs[index]
            index += 1

        while not outputs[index].startswith("Completed"):
            index += 1

        timing = outputs[index].split()[-1]
        conversation["time_used_in_seconds"] = timing[:-1]

        result.append(conversation)

    index += 1
    total_time_used = outputs[index].split()[-2]

    index += 1
    num_tokens_generated = outputs[index].split()[-1]

    index += 1
    token_per_second = outputs[index].split()[-2]

    return {
        "conversations": result,
        "total_time_used": total_time_used,
        "num_tokens_generated": num_tokens_generated,
        "token_per_second": token_per_second,
    }


def parse_results(notebook):
    result = {}
    repetition_penalty = None
    for x in notebook["cells"]:
        source = x["source"]
        for x2 in source:
            # print(x2)
            if "_RP" in x2:
                start = x2.index("1.")
                end = x2.index('"', start)
                repetition_penalty = x2[start:end]
                print("processing repetition_penalty:", repetition_penalty)

        if source and repetition_penalty:
            outputs = x["outputs"][0]["text"]
            result[repetition_penalty] = parse_outputs(outputs)
            repetition_penalty = None

    return result


def calc_ragas_scores(conversations):
    dict = {
        "question": [],
        "user_question": [],
        "standalone_question": [],
        "contexts": [],
        "answer": [],
    }

    for conversation in conversations:
        standalone_question = (
            conversation["standalone_question"]
            if "standalone_question" in conversation
            else conversation["question"]
        )
        dict["question"].append(standalone_question)
        dict["answer"].append(conversation["answer"])

        dict["user_question"].append(conversation["question"])
        dict["standalone_question"].append(
            conversation["standalone_question"]
            if "standalone_question" in conversation
            else ""
        )

        contexts = []
        docs = qa.retriever.get_relevant_documents(standalone_question)
        for doc in docs:
            contexts.append(doc.page_content)

        dict["contexts"].append(contexts)

    # print(dict)

    ds = Dataset.from_dict(dict)

    result = evaluate(
        ds,
        metrics=[
            faithfulness,
            answer_relevancy,
        ],
        llm=gpt4_turbo,
    )

    result["overall_score"] = 2 / (
        1 / result["faithfulness"] + 1 / result["answer_relevancy"]
    )

    print(f"\n\n# Ragas scores: {result}\n")
    return dict, result


def evaluate_models(model_names, prefix="nvidia-4090"):
    raw_data = {
        "model_name": [],
        "repetition_penalty": [],
        "user_question": [],
        "standalone_question": [],
        "contexts": [],
        "answer": [],
    }
    perf_data = {
        "model_name": [],
        "repetition_penalty": [],
        "faithfulness": [],
        "answer_relevancy": [],
        "overall_score": [],
        "total_time_used": [],
        "num_tokens_generated": [],
        "token_per_second": [],
    }

    repetition_penalties = ["1.05", "1.10", "1.15"]

    openai_model_names = {
        "1.05": "gpt-3.5-turbo",
        "1.10": "gpt-3.5-turbo-instruct",
        "1.15": "gpt-4",
    }

    for model_name in model_names:
        notebook = load_notebook(f"./notebook/{prefix}-{model_name}.ipynb")
        results = parse_results(notebook)
        for repetition_penalty in repetition_penalties:
            result = results[repetition_penalty]
            dict, ragas = calc_ragas_scores(result["conversations"])

            if model_name == "openai" or model_name.startswith("gpt-"):
                model_name = openai_model_names[repetition_penalty]
                repetition_penalty = ""

            for _ in dict["question"]:
                raw_data["model_name"].append(model_name)
                raw_data["repetition_penalty"].append(repetition_penalty)

            raw_data["user_question"] += dict["user_question"]
            raw_data["standalone_question"] += dict["standalone_question"]
            raw_data["contexts"] += dict["contexts"]
            raw_data["answer"] += dict["answer"]

            perf_data["model_name"].append(model_name)
            perf_data["repetition_penalty"].append(repetition_penalty)

            perf_data["faithfulness"].append(ragas["faithfulness"])
            perf_data["answer_relevancy"].append(ragas["answer_relevancy"])
            perf_data["overall_score"].append(ragas["overall_score"])
            perf_data["num_tokens_generated"].append(
                int(result["num_tokens_generated"])
            )
            perf_data["total_time_used"].append(float(result["total_time_used"]))
            perf_data["token_per_second"].append(float(result["token_per_second"]))

    perf_ds = Dataset.from_dict(perf_data)
    perf_pd = perf_ds.to_pandas()

    raw_ds = Dataset.from_dict(raw_data)
    raw_pd = raw_ds.to_pandas()
    return perf_pd, raw_pd
