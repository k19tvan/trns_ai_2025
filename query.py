from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from retrieval import get_retrievals

def main():

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question, also giving explanation.
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "context"],
    )
    
    question = "I have been attending all my lectures regularly, but due to a personal emergency, I missed most of my lab sessions. I completed all assignments and scored well in my exams. Will I still pass the course?"
    
    retrievals = get_retrievals(question, 1)
    
    print(retrievals)
    
    context = "\n".join([doc.page_content for doc in retrievals])
    
    llm = ChatOllama(model = "tinyllama", temperature=0.00)
    pipeline = prompt | llm
    
    answer = pipeline.invoke({"question" : question, "context" : context})
    
    print(answer)
    
if __name__ == "__main__": main()
    