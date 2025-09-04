# from langchain_huggingface.chat_models import ChatHuggingFace
# from langchain_huggingface import HuggingFaceEndpoint
# from dotenv import load_dotenv
# import os

# # Load .env variables
# load_dotenv()

# # Initialize Hugging Face endpoint with your API token
# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-alpha",  # Open-source generative model
#     task="text-generation",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
# )

# # Wrap endpoint in ChatHuggingFace for easier use
# model = ChatHuggingFace(llm=llm)

# # Example usage
# if __name__ == "__main__":
#     question = "What is the maximum working hours per week according to Saudi labor law?"
#     answer = model.invoke(question)
#     print(answer)


from langchain_community.chat_models import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:9090/v1",
    api_key="EMPTY",   # vLLM doesn’t require auth
    model="my-model",  # your served model name
)

# Optional: wrap in a helper object for RAG or direct usage
model = llm
if __name__ == "__main__":
    question = "What is the maximum working hours per week according to Saudi labor law?"
    answer = model.invoke(question)
    print(answer)