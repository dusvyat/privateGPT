
from langchain.chains import RetrievalQA
from settings import load_llm, load_chroma


def run():

    # Load the language model
    llm = load_llm()
    db = load_chroma()
    # Create the retriever
    retriever = db.as_retriever()

    # Create the chain and retrieve relevant text chunks, only use the relevant text chunks in the language model
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        
        # Get the answer from the chain
        res = qa(query)    
        answer, docs = res['result'], res['source_documents']

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)
        
        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)


if __name__ == "__main__":
    run()
