# llm/GroqClient.py

import os
import time

from groq import Groq
from loguru import logger

from config import (
    GROQ_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    SYSTEM_PROMPT,
)

class GroqClient:

    def __init__(self):

        api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not api_key:

            raise EnvironmentError(
                "GROQ_API_KEY not set."
            )

        self.client = Groq(
            api_key=api_key
        )

        self.model = GROQ_MODEL

        self._verify()

    # =====================================================
    # VERIFY
    # =====================================================

    def _verify(self):

        try:

            self.client.chat.completions.create(

                model=self.model,

                messages=[
                    {
                        "role": "user",
                        "content": "hello"
                    }
                ],

                max_tokens=5
            )

            logger.info(
                f"Groq API ready — {self.model}"
            )

        except Exception as e:

            raise EnvironmentError(
                f"GROQ_API_KEY invalid.\n"
                f"Error: {e}"
            )

    # =====================================================
    # GENERATE
    # =====================================================

    def generate(

        self,

        prompt: str,

        stream: bool = False

    ) -> str:

        try:

            t0 = time.time()

            response = self.client.chat.completions.create(

                model=self.model,

                messages=[

                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },

                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=LLM_TEMPERATURE,

                max_tokens=LLM_MAX_TOKENS,
            )

            answer = (
                response.choices[0]
                .message.content
                .strip()
            )

            logger.info(
                f"Generation completed in "
                f"{round(time.time()-t0,2)}s"
            )

            return answer

        except Exception as e:

            logger.error(
                f"Generation failed: {e}"
            )

            return (
                f"Generation error: {e}"
            )