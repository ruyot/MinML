"""
FastAPI middleware for semantic compression service.
"""

from .main import app
from .models import *
from .compression_pipeline import *

__all__ = ['app'] 