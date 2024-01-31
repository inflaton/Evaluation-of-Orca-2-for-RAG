from langchain.chains import ConversationalRetrievalChain
from langchain.chains.base import Chain

from app_modules.llm_inference import LLMInference


class QAChain(LLMInference):
    def __init__(self, vectorstore, llm_loader):
        super().__init__(llm_loader)
        self.vectorstore = vectorstore

    def create_chain(self) -> Chain:
        qa = ConversationalRetrievalChain.from_llm(
            self.llm_loader.llm,
            self.vectorstore.as_retriever(search_kwargs=self.llm_loader.search_kwargs),
            max_tokens_limit=self.llm_loader.max_tokens_limit,
            return_source_documents=True,
        )

        return qa
