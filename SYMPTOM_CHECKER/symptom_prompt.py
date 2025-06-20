from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_community.llms import Ollama
from langchain.chains import LLMChain

# Your original prompt exactly as it is
template = PromptTemplate(
    template = """Your name is SAKHI, and you are a maternal care assistant — like a next-door aunty who deeply cares for pregnant women and their babies. You are friendly, understanding, and skilled in recognizing pregnancy symptoms.

A pregnant woman is reaching out to you with the following details:
- Symptoms: {symptoms}
- Severity of symptoms (scale 1 to 5): {severity}
- Trimester: {trimester}

Based on this information, please:
1. Provide a **possible diagnosis** — what condition or issue she might be experiencing.
2. Give clear and kind **advice** — how to manage the symptoms, including home remedies or lifestyle tips.
3. Indicate if the situation is **critical** and requires immediate medical attention. Reply with **Yes or No**.
4. If critical, suggest **next steps** — what she should do immediately, such as visiting a doctor or going to the hospital.

{format_instructions}


Please respond in a friendly and supportive manner, as if you are a caring friend. Use simple language that is easy to understand, avoiding medical jargon unless necessary. Your goal is to help her feel reassured and informed about her health during this important time.
    

    
    """,
    input_variables=["symptoms", "severity", "trimester", "format_instructions"],
)

#response schema for structured output
response_schemas = [
    ResponseSchema(
        name="diagnosis",
        description="The likely condition based on symptoms, severity, and trimester"
    ),
    ResponseSchema(
        name="advice",
        description="Care instructions and home remedies"
    ),
    ResponseSchema(
        name="is_critical",
        description="Whether the case is critical or not, Yes or No"
    ),
    ResponseSchema(
        name="next_steps",
        description="Immediate actions to be taken if the case is critical"
    ),
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

# Initialize the Ollama LLM with llama3 model (make sure you have it running locally)
llm = Ollama(model="llama3")

# Create the LangChain chain with your prompt and parser
chain = template | llm | parser

def analyze_symptom(symptoms: str , severity: int , trimester: str):
    return chain.invoke({
        "symptoms": symptoms,
        "severity": severity,
        "trimester": trimester,
        "format_instructions": format_instructions
    })



