import langchain
from langchain.prompts import PromptTemplate
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

tourist_guide_pakistan_template = """
You are a tourist guide chatbot specialized in providing information about tourist destinations in Pakistan.
Your expertise lies in offering details about places to visit, cultural insights, historical landmarks, and travel tips within Pakistan.
Please note that your knowledge is limited to Pakistan's tourist domain,
and you won't entertain queries beyond this scope. If asked about topics unrelated to Pakistani tourism,
kindly apologize and redirect the conversation back to tourist-related inquiries.

Question: {question}
Answer:"""

tourist_guide_prompt = PromptTemplate(
    input_variables=["question"],
    template=tourist_guide_pakistan_template
)

text = "Which are the top 3 places to visit in pakistan?"
print(tourist_guide_prompt.format(question=text))


MODEL_NAME = "TheBloke/Llama-2-13b-Chat-GPTQ"
 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
 
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto"
)
 
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.0001
generation_config.top_p = 0.95
generation_config.do_sample = True
generation_config.repetition_penalty = 1.15
 
text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    generation_config=generation_config,
)
 
llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})