"""FastAPI package — chat SSE, PDF serve, citation lookup."""

__all__ = ["app"]


def __getattr__(name: str):
    if name == "app":
        from api.main import app

        return app
    raise AttributeError(name)
