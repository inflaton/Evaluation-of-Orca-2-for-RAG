from langchain.chains import ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain.memory import ConversationSummaryBufferMemory

from app_modules.llm_inference import LLMInference


class QAChain(LLMInference):
    def __init__(self, vectorstore, llm_loader):
        super().__init__(llm_loader)
        self.vectorstore = vectorstore

    def create_chain(self) -> Chain:
        memory = ConversationSummaryBufferMemory(
            llm=self.llm_loader.llm,
            output_key="answer",
            memory_key="chat_history",
            max_token_limit=1024,
            return_messages=True,
        )
        qa = ConversationalRetrievalChain.from_llm(
            self.llm_loader.llm,
            memory=memory,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs=self.llm_loader.search_kwargs
            ),
            get_chat_history=lambda h: h,
            return_source_documents=True,
        )

        return qa
