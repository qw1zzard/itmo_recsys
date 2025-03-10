from fastapi import FastAPI

from service.api.exception_handlers import add_exception_handlers
from service.api.middlewares import add_middlewares
from service.api.views import add_views
from service.log import setup_logging
from service.settings import ServiceConfig

__all__ = ("create_app",)


def create_app(config: ServiceConfig) -> FastAPI:
    setup_logging(config)

    app = FastAPI(debug=False)
    app.state.k_recs = config.k_recs

    add_views(app)
    add_middlewares(app)
    add_exception_handlers(app)

    return app
