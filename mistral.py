# from langchain_community.llms import mistral
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

bisimo = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(bisimo)
model = AutoModelForCausalLM.from_pretrained(bisimo, device_map="auto")

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=pipe)

# load_dotenv()
# api_key = os.getenv("MISTRAL_API_KEY")

# llm = mistral(model_kwargs={'api_key': api_key}, temperature=0)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

while True:
    user_input = input("Kamu: ")
    response = conversation.run(input=user_input)
    print("Chatbot:", response)