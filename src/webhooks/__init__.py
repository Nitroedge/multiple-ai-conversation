"""
Webhook endpoints for n8n integration and external services
"""

from .router import webhook_router

__all__ = ["webhook_router"]