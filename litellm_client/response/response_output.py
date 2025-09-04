class ResponseOutput:
    def __init__(self, raw_response):
        self.raw = raw_response

        if hasattr(raw_response, "choices") and raw_response.choices:
            self.text = raw_response.choices[0].message.content
        else:
            self.text = ""

    def transform(self) -> str:
        return self.text

    def usage(self) -> dict:
        return getattr(self.raw, "usage", None)
