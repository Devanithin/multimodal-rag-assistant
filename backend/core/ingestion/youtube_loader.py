from langchain_community.document_loaders import YoutubeLoader


def parse_youtube_url(url: str) -> str:
    loader = YoutubeLoader.from_youtube_url(
        url,
        add_video_info=False,
        language=["en", "hi"],
    )
    docs = loader.load()
    text = " ".join([doc.page_content for doc in docs])
    return text.strip()