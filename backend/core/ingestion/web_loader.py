from langchain_community.document_loaders import WebBaseLoader


def parse_web_url(url: str) -> str:
    loader = WebBaseLoader(url)
    docs = loader.load()
    text = " ".join([doc.page_content for doc in docs])
    return text.strip()