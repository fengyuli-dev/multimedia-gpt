import argparse
import os
import re
import shutil
import uuid

import gradio as gr
import numpy as np
import openai
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from PIL import Image

from models import *
from utils import *

openai.api_key = os.environ["OPENAI_API_KEY"]


PROMPT_PREFIX = """Multimedia GPT is designed to be able to assist with a wide range of documents, visual, and audio related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. Multimedia GPT is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Multimedia GPT is able to process and understand large amounts of documents, images, videos, and audio recordings. As a language model, Multimedia GPT can not directly read images, videos, or audio recordings, but it has a list of tools to finish different visual / audio tasks. Each media will have a file name, and Multimedia GPT can invoke different tools to indirectly understand pictures. When talking about images, Multimedia GPT is very strict to the file name and will never fabricate nonexistent files. When using tools to generate new image files, Multimedia GPT is also known that the image may not be the same as the user's demand, and will use other visual question answering tools or description tools to observe the real image. Multimedia GPT is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the content and the file name. It will remember to provide the file name from the last tool observation, if a new file is generated.

Human may provide new images, audio recordings, videos, or other to Multimedia GPT with a description. The description helps Multimedia GPT to understand this file, but Multimedia GPT should use tools to finish following tasks, rather than directly imagine from the description.

Overall, Multimedia GPT is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 


TOOLS:
------

Multimedia GPT  has access to the following tools:"""

FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

PROMPT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since Multimedia GPT is a text language model, Multimedia GPT must use tools to observe images, videos, and audio recodings, rather than imagination.
The thoughts and observations are only visible for Multimedia GPT, Multimedia GPT should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad}"""

os.makedirs("image", exist_ok=True)
os.makedirs("audio", exist_ok=True)
os.makedirs("video", exist_ok=True)


class ConversationBot:
    def __init__(self, load_dict, llm):
        # load_dict = {'VisualQuestionAnswering':'cuda:0', 'ImageCaptioning':'cuda:1', ...}
        print(f"Initializing Multimedia GPT, load_dict={load_dict}")
        if "ImageCaptioning" not in load_dict:
            raise ValueError(
                "You have to load ImageCaptioning as a basic function for Multimedia GPT"
            )

        self.llm = OpenAI(model_name=llm, temperature=0, openai_api_key=openai.api_key)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", output_key="output"
        )

        self.models = dict()
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)

        self.tools = []
        for class_name, instance in self.models.items():
            for e in dir(instance):
                if e.startswith("inference"):
                    func = getattr(instance, e)
                    self.tools.append(
                        Tool(name=func.name, description=func.description, func=func)
                    )

        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={
                "prefix": PROMPT_PREFIX,
                "format_instructions": FORMAT_INSTRUCTIONS,
                "suffix": PROMPT_SUFFIX,
            },
        )

    def run_text(self, text, state):
        self.agent.memory.buffer = cut_dialogue_history(
            self.agent.memory.buffer, keep_last_n_words=500
        )
        res = self.agent({"input": text})
        res["output"] = res["output"].replace("\\", "/")
        response = re.sub(
            "(image/\S*png)",
            lambda m: f"![](/file={m.group(0)})*{m.group(0)}*",
            res["output"],
        )
        state = state + [(text, response)]
        print(
            f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
            f"Current Memory: {self.agent.memory.buffer}"
        )
        return state, state

    def run_image(self, image, state, txt):
        image_filename = os.path.join("image", str(uuid.uuid4())[0:8] + ".png")
        print("======>Auto Resize Image...")
        image_path = image.name
        img = Image.open(image_path)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        width_new = int(np.round(width_new / 64.0)) * 64
        height_new = int(np.round(height_new / 64.0)) * 64
        img = img.resize((width_new, height_new))
        img = img.convert("RGB")
        img.save(image_filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        description = self.models["ImageCaptioning"].inference(image_filename)
        Human_prompt = (
            "\nHuman: provide a figure named {}. The description is: {}. "
            "This information helps you to understand this image, "
            "but you should use tools to finish following tasks, "
            'rather than directly imagine from my description. If you understand, say "Image file received". \n'.format(
                image_filename, description
            )
        )
        AI_prompt = "Image file received.  "
        self.agent.memory.buffer = (
            self.agent.memory.buffer + Human_prompt + "AI: " + AI_prompt
        )
        state = state + [(f"![](/file={image_filename})*{image_filename}*", AI_prompt)]
        print(
            f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: {state}\n"
            f"Current Memory: {self.agent.memory.buffer}"
        )
        return state, state, txt + " " + image_filename + " "

    def run_audio(self, audio, state, txt):
        audio_filename = os.path.join("audio", str(uuid.uuid4())[0:8] + ".mp3")
        audio_path = audio.name
        shutil.copy(audio_path, audio_filename)
        description = self.models["Whisper"].inference(audio_filename)
        Human_prompt = (
            "\nHuman: provide a audio recording named {}. The description is: {}. "
            "This information helps you to understand this audio recording, "
            "but you should use tools to finish following tasks, "
            'rather than directly imagine from my description. If you understand, say "Audio file received". \n'.format(
                audio_filename, description
            )
        )
        AI_prompt = "Audio file received.  "
        self.agent.memory.buffer = (
            self.agent.memory.buffer + Human_prompt + "AI: " + AI_prompt
        )
        state = state + [(f"![](/file={audio_filename})*{audio_filename}*", AI_prompt)]
        print(
            f"\nProcessed run_audio, Input audio: {audio_filename}\nCurrent state: {state}\n"
            f"Current Memory: {self.agent.memory.buffer}"
        )
        return state, state, txt + " " + audio_filename + " "

    def run_video(self, video, state, txt):
        raise NotImplementedError

    def run_pdf(self, pdf, state, txt):
        pdf_path = pdf.name
        pdf_filename = os.path.join("audio", str(uuid.uuid4())[0:8] + ".pdf")
        shutil.copy(pdf_path, pdf_filename)
        PDFReader = globals()["PDFReader"](device="cpu")
        PDFReader.init_index(pdf_path)
        Human_prompt = (
            "\nHuman: provide a pdf document named {}. The description is stored as an index in your PDFReader. "
            "You can find the content of the pdf document by using the tool PDFReader later"
            "This information helps you to understand this pdf document, "
            "but you should use tools to finish following tasks, "
            'rather than directly imagine from my description. If you understand, say "PDF file received". \n'.format(
                pdf_filename
            )
        )
        AI_prompt = "PDF file received."
        self.agent.memory.buffer = (
            self.agent.memory.buffer + Human_prompt + "AI: " + AI_prompt
        )
        state = state + [(f"![](/file={pdf_filename})*{pdf_filename}*", AI_prompt)]
        print(
            f"\nProcessed run_pdf, Input pdf: {pdf_filename}\nCurrent state: {state}\n"
            f"Current Memory: {self.agent.memory.buffer}"
        )
        return state, state, txt + " " + pdf_filename + " "

    def run_multimedia(self, file, state, txt):
        print(f"Original path to the uploaded file is {file.name}")
        ext = os.path.splitext(file.name)[1]
        if ext in [".png", ".jpg", ".jpeg"]:
            return self.run_image(file, state, txt)
        elif ext in [".mp3", ".wav", ".m4a"]:
            return self.run_audio(file, state, txt)
        elif ext in ["mp4"]:
            return self.run_video(file, state, txt)
        elif ext in [".pdf"]:
            return self.run_pdf(file, state, txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load",
        type=str,
        default="ImageCaptioning_cpu,DALLE_cpu,Whisper_cpu",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="text-davinci-003",
        choices=["text-davinci-003", "gpt-3.5-turbo", "gpt-4"],
    )
    args = parser.parse_args()
    llm = args.llm.strip()
    load_dict = {
        e.split("_")[0].strip(): e.split("_")[1].strip() for e in args.load.split(",")
    }
    bot = ConversationBot(load_dict=load_dict, llm=llm)
    with gr.Blocks(css="#chatbot .gradio-container") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot", label="Multimedia GPT")
        state = gr.State([])
        with gr.Row():
            with gr.Column(scale=0.7):
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press enter, or upload a file",
                ).style(container=False)
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button("Clear")
            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton("Upload", file_types=["file"])

        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        btn.upload(bot.run_multimedia, [btn, state, txt], [chatbot, state, txt])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
        demo.launch(inbrowser=True, server_name="0.0.0.0", server_port=5050)
